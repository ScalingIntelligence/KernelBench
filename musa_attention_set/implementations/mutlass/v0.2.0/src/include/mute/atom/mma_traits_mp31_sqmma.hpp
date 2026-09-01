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

#include <mute/arch/mma_mp31.hpp>
#include <mute/atom/mma_traits.hpp>

#include <mute/tensor.hpp>

namespace mute {

namespace MP31::SQMMA {

////////////////////////////////////////////
// Common layouts for SQMMA Shared Memory //
////////////////////////////////////////////
template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_Atom_Bits = mute::conditional_t<tnsp == TCE::Major::MN,
                            decltype(make_layout(Shape<Int<MN * sizeof_bits_v<Type>>, Int<K>>{}, LayoutLeft{})),
                            decltype(make_layout(Shape<Int<MN>, Int<K * sizeof_bits_v<Type>>>{}, LayoutRight{}))>;

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL256_SS256_NONE_Atom = decltype(upcast<sizeof_bits_v<Type>>(ComposedLayout<Swizzle<0,4,4>, smem_ptr_flag, Layout_Atom_Bits<MN,K,Type,tnsp>>{}));

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL256_SS256_SG16_Atom = decltype(upcast<sizeof_bits_v<Type>>(ComposedLayout<Swizzle<4,4,4>, smem_ptr_flag, Layout_Atom_Bits<MN,K,Type,tnsp>>{}));

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL256_SS256_SG32_Atom = decltype(upcast<sizeof_bits_v<Type>>(ComposedLayout<Swizzle<3,5,3>, smem_ptr_flag, Layout_Atom_Bits<MN,K,Type,tnsp>>{}));

template <int MN, int K, class Type, TCE::Major tnsp>
using Layout_SL256_SS256_SG64_Atom = decltype(upcast<sizeof_bits_v<Type>>(ComposedLayout<Swizzle<2,6,2>, smem_ptr_flag, Layout_Atom_Bits<MN,K,Type,tnsp>>{}));

// Only used for canonical gemm
template <class Type, TCE::Major tnsp, class AtomMN, class AtomK>
MUTE_HOST_DEVICE constexpr
auto
make_canonical_gemm_smem_atom_layout() {
  constexpr int BITS = sizeof_bits_v<Type>;
  constexpr int ATOM_MN = size(AtomMN{});
  constexpr int ATOM_K  = size(AtomK{});
  if constexpr (tnsp == TCE::Major::MN) {
    // S4/S8/U8/FP8
    if constexpr (BITS <= 8) {
      return Layout_SL256_SS256_SG16_Atom<ATOM_MN, ATOM_K, Type, tnsp>{};
    }
    // FP16/BF16
    else if constexpr (BITS == 16) {
      return Layout_SL256_SS256_SG32_Atom<ATOM_MN, ATOM_K, Type, tnsp>{};
    }
    // TF32
    else if constexpr (BITS == 32) {
      return Layout_SL256_SS256_SG64_Atom<ATOM_MN, ATOM_K, Type, tnsp>{};
    }
    else {
      static_assert(BITS <= 32, "Unsupported Sqmma Type");
    }
  }
  else if constexpr (tnsp == TCE::Major::K) {
    return Layout_SL256_SS256_SG16_Atom<ATOM_MN, ATOM_K, Type, tnsp>{};
  }
  else {
    static_assert(tnsp != TCE::Major::MN && tnsp != TCE::Major::K, "Unrecognized MajorMode!");
  }
}

//
// Tensor (position-dependent swizzle) to SwizzleGranularityType utility
//

template <class Engine, class Shape, class Stride>
MUTE_HOST_DEVICE constexpr
SwizzleGranularityType
swizzle_granularity_type(Tensor<Engine, Layout<Shape, Stride>> const&)
{
  static_assert(is_same<uint8_t, typename Engine::value_type>::value,
                "Expected uint8_t type in SwizzleGranularityType conversion.");

  using Swizzle = get_swizzle_t<Engine>;
  constexpr int B = Swizzle::num_bits;
  constexpr int M = Swizzle::num_base;
  constexpr int S = Swizzle::num_shft;

  static_assert(4 <= M && M <= 6, "Unsupported swizzle granularity");

  // The Swizzle Line for Sqmma is 256B(2^8)
  static_assert((M+S) == 8, "Unsupported swizzle granularity");

  if constexpr (B == 0) {
    return SwizzleGranularityType::SWIZZLE_GRANULARITY_NONE;
  }

  // The Swizzle Stride for Mp31 Sqmma must be 256B(2^8)
  static_assert((M+B) == 8, "Unsupported swizzle granularity");

  switch (M) {
    case 4: return SwizzleGranularityType::SWIZZLE_GRANULARITY_B16;
    case 5: return SwizzleGranularityType::SWIZZLE_GRANULARITY_B32;
    case 6: return SwizzleGranularityType::SWIZZLE_GRANULARITY_B64;
  }
  return SwizzleGranularityType::SWIZZLE_GRANULARITY_NONE;
}


///////////////////////////////////////////////////////////////////////////////
// Construction method for SQMMA Descriptors
///////////////////////////////////////////////////////////////////////////////
template <TCE::Major MajorMode, class TEngine, class TLayout>
MUTE_HOST_DEVICE constexpr
SqmmaDescriptor
make_sqmma_desc(Tensor<TEngine, TLayout> const& tensor)
{
  static_assert(is_smem<TEngine>::value, "SQMMA Descriptors can only be constructed on smem.");
  static_assert(TLayout::rank == 2, "SQMMA Descriptors can obly be constructed on rank-2 tensors.");
  using value_type = typename TEngine::value_type;

  Tensor u8_tensor = recast<uint8_t const>(tensor);

  // Result
  SqmmaDescriptor desc;

  // SwizzleGranularity type
  constexpr SQMMA::SwizzleGranularityType SG_TYPE = SQMMA::swizzle_granularity_type(u8_tensor);
  desc.bitfield.swizzle_granularity_type_ = static_cast<uint8_t>(SG_TYPE);

  // Start address
  desc.bitfield.start_address_ = cast_smem_ptr_to_uint(raw_pointer_cast(u8_tensor.data()));

  if constexpr (MajorMode == TCE::Major::MN)
  {
    MUTE_STATIC_ASSERT_V(size<0>(u8_tensor) % Int<16>{} == Int<0>{},    // M|N  size
                         "Not a canonical SQMMA_MN Layout: Expected MN-size multiple of 16.");

    MUTE_STATIC_ASSERT_V(size<1>(u8_tensor) % Int<8>{} == Int<0>{} &&
                         size<1>(u8_tensor) <= Int<256>{},                         // K    size
                         "Not a canonical SQMMA_MN Layout: Expected K-size multiple of 8 (in units of uint8_t).");

    constexpr uint32_t stride_1 = stride<1>(u8_tensor);
    static_assert(8 <= stride_1 && stride_1 <= 256, "Not a canonical SQMMA_MN Layout: Expected stride failure.");

    switch (stride_1) {
      case   8: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B8);   break;
      case  16: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B16);  break;
      case  32: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B32);  break;
      case  64: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B64);  break;
      case 128: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B128); break;
      case 256: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B256); break;
    }
  }
  else if constexpr (MajorMode == TCE::Major::K)
  {
    MUTE_STATIC_ASSERT_V(size<0>(u8_tensor) % Int<16>{} == Int<0>{},    // M|N  size
                         "Not a canonical SQMMA_K Layout: Expected MN-size multiple of 16.");
    MUTE_STATIC_ASSERT_V(size<1>(u8_tensor) % Int<8>{} == Int<0>{} &&
                         size<1>(u8_tensor) <= Int<256>{},                         // K    size
                         "Not a canonical SQMMA_K Layout: Expected K-size multiple of 8 (in units of uint8_t).");

    constexpr uint32_t stride_0 = stride<0>(u8_tensor);
    static_assert(8 <= stride_0 && stride_0 <= 256, "Not a canonical SQMMA_K Layout: Expected stride failure.");

    switch (stride_0) {
      case   8: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B8);   break;
      case  16: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B16);  break;
      case  32: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B32);  break;
      case  64: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B64);  break;
      case 128: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B128); break;
      case 256: desc.bitfield.leading_stride_type_ = static_cast<uint8_t>(LeadingStrideType::B256); break;
    }
  }
  else {
    static_assert(MajorMode != TCE::Major::MN && MajorMode != TCE::Major::K, "Unrecognized MajorMode!");
  }
  return desc;
}

///////////////////////////////////////////////////////////////////////////////
// Higher level SQMMA Descriptor utilities
///////////////////////////////////////////////////////////////////////////////

struct DescriptorIterator
{
  using reference    = SqmmaDescriptor;
  using element_type = SqmmaDescriptor;
  using value_type   = SqmmaDescriptor;

  SqmmaDescriptor desc_;

  // Dereference returns the SqmmaDescriptor
  MUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  // Advance and return a new SqmmaDescriptor
  template <class Index>
  MUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  MUTE_HOST_DEVICE constexpr
  DescriptorIterator operator+(Index const& offset) const
  {
    return { SqmmaDescriptor{desc_ + uint64_t(offset)} };
  }

  MUTE_HOST_DEVICE constexpr
  operator int32_t() const { return static_cast<int32_t>(desc_); }

  MUTE_HOST_DEVICE friend void
  print(DescriptorIterator) { printf("SQMMA::DescriptorIterator"); }
};

MUTE_HOST_DEVICE constexpr
SqmmaDescriptor
raw_pointer_cast(DescriptorIterator const& ptr) {
  return ptr.desc_;
}

// Recast a DescriptorIterator Tensor to uint32_t, it's RegType in mma_unpack
template <class NewT>
MUTE_HOST_DEVICE constexpr
DescriptorIterator
recast_ptr(DescriptorIterator const& iter) {
  static_assert(is_same<NewT, int32_t>::value, "Can only cast SqmmaDescriptorIterator to int32_t.");
  return iter;
}

// The SQMMA Traits below have custom fragment type flags for their smem desc tensors.
// These flags specialize a MakeTensor customization point to correctly make the fragment that is desired.
template <TCE::Major>
struct smem_desc : DescriptorIterator {};

} // namespace MP31::SQMMA

// Customization point for creating a SQMMA::smem_desc Tensor
template <TCE::Major MajorMode>
struct MakeTensor<MP31::SQMMA::smem_desc<MajorMode>>
{
  template <class TEngine, class TLayout>
  MUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a SQMMA Desc Tensor");
    return make_tensor(MP31::SQMMA::DescriptorIterator{MP31::SQMMA::make_sqmma_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint8_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};

///////////////////////////////////////////////////////////////////////////////
//////////////////////////// MMA_TRAITS ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace MP31::SQMMA {

// Accumulator layouts
template <int M, int N>
using CLayout = Layout<Shape <Shape <    _8, _4, _4>, Shape <      _2, Int<N/16>, Int<M/16>>>,
                       Stride<Stride<Int<M>, _1, _4>, Stride<Int<M*8>, Int<M*16>,      _16>>>;

// Shared memory source layouts for any value type
template <int M, int K>
using ABLayout  = Layout<Shape <_128, Shape <Int<M>,Int<K>>>,
                         Stride<  _0, Stride<    _1,Int<M>>>>;

} // namespace MP31::SQMMA

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  U32 +=   U8 *   U8 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x64x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x64x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x64x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x32x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x32x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x32x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x64x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x64x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x64x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x128x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x128x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x128x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x16x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x16x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x16x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x32x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x32x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x32x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x64x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x64x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x64x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x128x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x128x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x128x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x32x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x32x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x32x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x64x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x64x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x64x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x128x32_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x128x64_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x128x128_U32U8U8_SS<tnspA, tnspB>>
{
  using ValTypeD = uint32_t;
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeC = uint32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  S32 +=   S8 *   S8 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x64x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x64x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_16x64x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x32x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x32x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x32x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x64x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x64x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x64x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x128x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x128x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_32x128x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x16x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x16x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x16x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x32x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x32x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x32x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x64x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x64x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x64x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x128x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x128x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_64x128x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x32x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x32x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x32x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x64x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x64x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x64x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x128x32_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x128x64_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB>
struct MMA_Traits<MP31_128x128x128_S32S8S8_SS<tnspA, tnspB>>
{
  using ValTypeD = int32_t;
  using ValTypeA = int8_t;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 +=  F16 *  F16 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x16_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  16>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  16>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 += BF16 * BF16 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x16_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  16>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  16>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 += TF32 * TF32 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x8_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _8>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,   8>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,   8>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x16_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x32_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x8_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _8>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,   8>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,   8>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x16_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x8_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _8>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,   8>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,   8>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x16_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x8_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _8>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,   8>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,   8>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x16_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x8_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _8>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,   8>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,   8>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x16_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x8_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _8>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,   8>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,   8>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x16_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x8_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _8>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,   8>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,   8>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x16_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  16>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  16>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x8_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _8>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,   8>;
  using BLayout   = MP31::SQMMA::ABLayout<128,   8>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x16_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _16>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  16>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  16>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32TF32TF32_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = tfloat32_t;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 += E4M3 * E4M3 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x128_F32E4M3E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 += E5M2 * E5M2 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x128_F32E5M2E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 +=  F16 *   S4 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_16x64x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_16x64x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x32x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x32x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x64x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x64x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x128x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x128x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x32x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x32x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x64x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x64x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x128x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x128x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x32x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x32x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x64x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x64x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x128x32_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x128x64_F32F16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 +=  F16 *   S8 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_16x64x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_16x64x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x32x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x32x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x64x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x64x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x128x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x128x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x32x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x32x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x64x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x64x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x128x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x128x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x32x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x32x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x64x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x64x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x128x32_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x128x64_F32F16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 +=   S4 *  F16 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32S4F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 +=   S8 *  F16 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32S8F16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 += E4M3 * E5M2 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x128_F32E4M3E5M2_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e4m3_t;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 += E5M2 * E4M3 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_16x64x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 16, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 32, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout< 64, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x128_F32E5M2E4M3_SS<tnspA, tnspB, scaleA, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = float_e5m2_t;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _128>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128, 128>;
  using BLayout   = MP31::SQMMA::ABLayout<128, 128>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 += BF16 *   S4 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_16x64x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_16x64x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x32x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x32x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x64x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x64x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x128x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x128x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x32x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x32x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x64x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x64x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x128x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x128x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x32x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x32x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x64x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x64x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x128x32_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x128x64_F32BF16S4_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int4_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 += BF16 *   S8 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M16N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_16x64x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_16x64x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 16,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x32x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x32x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x64x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x64x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x128x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_32x128x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x32x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x32x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x64x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x64x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x128x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_64x128x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x32x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x32x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x64x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x64x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x128x32_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleA>
struct MMA_Traits<MP31_128x128x64_F32BF16S8_SS<tnspA, tnspB, scaleA>>
{
  using ValTypeD = float;
  using ValTypeA = bfloat16_t;
  using ValTypeB = int8_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 +=   S4 * BF16 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32S4BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int4_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////  F32 +=   S8 * BF16 //////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

//
// M32N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x32x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x64x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M32N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_32x128x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_32, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 32, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N16
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x16x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _16, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 16,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  16>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x32x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x64x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M64N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_64x128x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout < 64, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N32
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x32x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _32, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 32,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  32>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N64
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x64x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _64, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout< 64,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128,  64>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};

//
// M128N128
//

template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x32_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _32>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  32>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  32>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


template <TCE::Major tnspA, TCE::Major tnspB, MP31::SQMMA::ScaleIn scaleB>
struct MMA_Traits<MP31_128x128x64_F32S8BF16_SS<tnspA, tnspB, scaleB>>
{
  using ValTypeD = float;
  using ValTypeA = int8_t;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = MP31::SQMMA::smem_desc<tnspA>;
  using FrgTypeB = MP31::SQMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_128, _128, _64>;
  using ThrID     = Layout<_128>;
  using ALayout   = MP31::SQMMA::ABLayout<128,  64>;
  using BLayout   = MP31::SQMMA::ABLayout<128,  64>;
  using CLayout   = MP31::SQMMA::CLayout <128, 128>;

  MP31::SQMMA::ScaleOut accumulate_ = MP31::SQMMA::ScaleOut::One;
};


} // namespace mute
