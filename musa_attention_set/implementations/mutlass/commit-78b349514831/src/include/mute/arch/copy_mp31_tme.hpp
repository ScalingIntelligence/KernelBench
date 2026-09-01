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

#include <mute/config.hpp>

#include <mute/arch/copy.hpp>
#include <mute/arch/copy_mp31.hpp>

namespace mute
{


////////////////////////////////////////////////////////////////////////////////////////////////////
/// TME_LOAD : Initiates a TME copy, in tile mode, from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP31_TME_LOAD {};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_1D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& dim0)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    __musa_tme_ld_tile_1d(bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim0, crd0,
                          static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl), static_cast<int32_t>(prefetch),
                          static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint), 0);
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }

  struct PREFETCH
  {
    MUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0,
         int32_t const& dim0)
    {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      __musa_tme_tile_prefetch_only_1d(gmem_int_desc, dim0, crd0);
#else
      MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
    }
  };
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_2D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& dim0,
       int32_t const& dim1)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v2i32_t crd {crd0, crd1};
    mute::v2i32_t dim {dim0, dim1};
    __musa_tme_ld_tile_2d(bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim, crd,
                          static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl), static_cast<int32_t>(prefetch),
                          static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint), 0);
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }

  struct PREFETCH
  {
    MUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0,
         int32_t const& crd1,
         int32_t const& dim0,
         int32_t const& dim1)
    {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      mute::v2i32_t crd {crd0, crd1};
      mute::v2i32_t dim {dim0, dim1};
      __musa_tme_tile_prefetch_only_2d(gmem_int_desc, dim, crd);
#else
      MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
    }
  };
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_3D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v3i32_t crd {crd0, crd1, crd2};
    mute::v3i32_t dim {dim0, dim1, dim2};
    __musa_tme_ld_tile_3d(bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim, crd,
                          static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl), static_cast<int32_t>(prefetch),
                          static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint), 0);
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }

  struct PREFETCH
  {
    MUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0,
         int32_t const& crd1,
         int32_t const& crd2,
         int32_t const& dim0,
         int32_t const& dim1,
         int32_t const& dim2)
    {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      mute::v3i32_t crd {crd0, crd1, crd2};
      mute::v3i32_t dim {dim0, dim1, dim2};
      __musa_tme_tile_prefetch_only_3d(gmem_int_desc, dim, crd);
#else
      MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
    }
  };
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_4D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v4i32_t crd {crd0, crd1, crd2, crd3};
    mute::v4i32_t dim {dim0, dim1, dim2, dim3};
    __musa_tme_ld_tile_4d(bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim, crd,
                          static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl), static_cast<int32_t>(prefetch),
                          static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint), 0);
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }

  struct PREFETCH
  {
    MUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0,
         int32_t const& crd1,
         int32_t const& crd2,
         int32_t const& crd3,
         int32_t const& dim0,
         int32_t const& dim1,
         int32_t const& dim2,
         int32_t const& dim3)
    {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      mute::v4i32_t crd {crd0, crd1, crd2, crd3};
      mute::v4i32_t dim {dim0, dim1, dim2, dim3};
      __musa_tme_tile_prefetch_only_4d(gmem_int_desc, dim, crd);
#else
      MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
    }
  };
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_5D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& crd4,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3,
       int32_t const& dim4)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v5i32_t crd {crd0, crd1, crd2, crd3, crd4};
    mute::v5i32_t dim {dim0, dim1, dim2, dim3, dim4};
    __musa_tme_ld_tile_5d(bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim, crd,
                          static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl), static_cast<int32_t>(prefetch),
                          static_cast<int32_t>(inner_hint), static_cast<int32_t>(outer_hint), 0);
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }

  struct PREFETCH
  {
    MUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0,
         int32_t const& crd1,
         int32_t const& crd2,
         int32_t const& crd3,
         int32_t const& crd4,
         int32_t const& dim0,
         int32_t const& dim1,
         int32_t const& dim2,
         int32_t const& dim3,
         int32_t const& dim4)
    {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      mute::v5i32_t crd {crd0, crd1, crd2, crd3, crd4};
      mute::v5i32_t dim {dim0, dim1, dim2, dim3, dim4};
      __musa_tme_tile_prefetch_only_5d(gmem_int_desc, dim, crd);
#else
      MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
    }
  };
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  TME::CacheHint      inner_hint,
  TME::CacheHint      outer_hint,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_ENTRY
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& dim0)
  {
    return MP31_TME_LOAD_1D<sg, ss, sl, inner_hint, outer_hint, prefetch>::copy(desc_ptr, bar_id, smem_ptr, crd0, dim0);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& dim0,
       int32_t const& dim1)
  {
    return MP31_TME_LOAD_2D<sg, ss, sl, inner_hint, outer_hint, prefetch>::copy(desc_ptr, bar_id, smem_ptr, crd0, crd1, dim0, dim1);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2)
  {
    return MP31_TME_LOAD_3D<sg, ss, sl, inner_hint, outer_hint, prefetch>::copy(desc_ptr, bar_id, smem_ptr, crd0, crd1, crd2, dim0, dim1, dim2);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3)
  {
    return MP31_TME_LOAD_4D<sg, ss, sl, inner_hint, outer_hint, prefetch>::copy(desc_ptr, bar_id, smem_ptr, crd0, crd1, crd2, crd3, dim0, dim1, dim2, dim3);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& crd4,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3,
       int32_t const& dim4)
  {
    return MP31_TME_LOAD_5D<sg, ss, sl, inner_hint, outer_hint, prefetch>::copy(desc_ptr, bar_id, smem_ptr, crd0, crd1, crd2, crd3, crd4, dim0, dim1, dim2, dim3, dim4);
  }

  struct PREFETCH;
};

struct MP31_TME_PREFETCH
{
  // doesn't matter for prefetch
  static constexpr TME::SmemSwizzleGranularity sg = TME::SmemSwizzleGranularity::NONE;
  static constexpr TME::SmemSwizzleStride      ss = TME::SmemSwizzleStride::B256;
  static constexpr TME::SmemSwizzleLine        sl = TME::SmemSwizzleLine::B256;
  static constexpr TME::CacheHint            hint = TME::CacheHint::CACHE_NORMAL;
  static constexpr PrefetchSize          prefetch = PrefetchSize::B128;

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       int32_t const& crd0,
       int32_t const& dim0)
  {
    return MP31_TME_LOAD_1D<sg, ss, sl, hint, hint, prefetch>::PREFETCH::copy(desc_ptr, crd0, dim0);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       int32_t const& crd0, int32_t const& crd1,
       int32_t const& dim0, int32_t const& dim1)
  {
    return MP31_TME_LOAD_2D<sg, ss, sl, hint, hint, prefetch>::PREFETCH::copy(desc_ptr, crd0, crd1, dim0, dim1);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2,
       int32_t const& dim0, int32_t const& dim1, int32_t const& dim2)
  {
    return MP31_TME_LOAD_3D<sg, ss, sl, hint, hint, prefetch>::PREFETCH::copy(desc_ptr, crd0, crd1, crd2, dim0, dim1, dim2);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3,
       int32_t const& dim0, int32_t const& dim1, int32_t const& dim2, int32_t const& dim3)
  {
    return MP31_TME_LOAD_4D<sg, ss, sl, hint, hint, prefetch>::PREFETCH::copy(desc_ptr, crd0, crd1, crd2, crd3, dim0, dim1, dim2, dim3);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4,
       int32_t const& dim0, int32_t const& dim1, int32_t const& dim2, int32_t const& dim3, int32_t const& dim4)
  {
    return MP31_TME_LOAD_5D<sg, ss, sl, hint, hint, prefetch>::PREFETCH::copy(desc_ptr, crd0, crd1, crd2, crd3, crd4, dim0, dim1, dim2, dim3, dim4);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// TME_LOAD_IM2COL : Initiates a TME copy, in im2col mode, from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

struct Im2ColWeightPos {
  int8_t s;
  int8_t r;
  int8_t t;
  int8_t reserve;

  MUTE_HOST_DEVICE constexpr
  operator int32_t() const {
    return static_cast<int32_t>(s) | (static_cast<int32_t>(r) << 8) | (static_cast<int32_t>(t) << 16);
  }
}; // struct Im2ColWeightPos

} // namespace detail

struct MP31_TME_LOAD_IM2COL {};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_IM2COL_3D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& coord_c,
       int32_t const& coord_q,
       int32_t const& coord_n,
       int8_t  const& weight_pos_s,
       int32_t const& range_c,
       int32_t const& range_nzpq,
       mute::v3i32_t const& conv_param,
       int32_t const& output_dim_q)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v3i32_t crd {coord_c, coord_q, coord_n};
    mute::v2i32_t blk_dim{range_c, range_nzpq};
    __musa_tme_ld_im2col_3d(bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, blk_dim, crd,
                            static_cast<int32_t>(weight_pos_s), output_dim_q, conv_param,static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl),
                            static_cast<int32_t>(prefetch));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }

  struct PREFETCH
  {
    MUTE_HOST_DEVICE static void
    copy(void    const* desc_ptr,
         int32_t const& coord_c,
         int32_t const& coord_q,
         int32_t const& coord_n,
         int32_t const& weight_pos_s,
         int32_t const& range_c,
         int32_t const& range_nzpq,
         mute::v3i32_t const& conv_param,
         int8_t  const& output_dim_q)
    {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      mute::v3i32_t crd {coord_c, coord_q, coord_n};
      mute::v2i32_t blk_dim{range_c, range_nzpq};
      __musa_tme_im2col_prefetch_only_3d(gmem_int_desc, blk_dim, crd, static_cast<int32_t>(weight_pos_s), output_dim_q, conv_param);
#else
      MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
    }
  };
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_IM2COL_4D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& coord_c,
       int32_t const& coord_q,
       int32_t const& coord_p,
       int32_t const& coord_n,
       int8_t  const& weight_pos_s,
       int8_t  const& weight_pos_r,
       int32_t const& range_c,
       int32_t const& range_nzpq,
       mute::v3i32_t const& conv_param,
       mute::v2i32_t const& output_dim)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v4i32_t crd {coord_c, coord_q, coord_p, coord_n};
    mute::v2i32_t blk_dim{range_c, range_nzpq};
    detail::Im2ColWeightPos weight_pos{weight_pos_s, weight_pos_r, 0};
    __musa_tme_ld_im2col_4d(bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, blk_dim, crd,
                            static_cast<int32_t>(weight_pos), output_dim, conv_param, static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl),
                            static_cast<int32_t>(prefetch));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }

  struct PREFETCH
  {
    MUTE_HOST_DEVICE static void
    copy(void    const* desc_ptr,
         int32_t const& coord_c,
         int32_t const& coord_q,
         int32_t const& coord_p,
         int32_t const& coord_n,
         int8_t  const& weight_pos_s,
         int8_t  const& weight_pos_r,
         int32_t const& range_c,
         int32_t const& range_nzpq,
         mute::v3i32_t const& conv_param,
         mute::v2i32_t const& output_dim)
    {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      mute::v4i32_t crd {coord_c, coord_q, coord_p, coord_n};
      mute::v2i32_t blk_dim{range_c, range_nzpq};
      detail::Im2ColWeightPos weight_pos{weight_pos_s, weight_pos_r, 0};
      __musa_tme_im2col_prefetch_only_4d(gmem_int_desc, blk_dim, crd, static_cast<int32_t>(weight_pos), output_dim, conv_param);
#else
      MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
    }
  };
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_IM2COL_5D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& coord_c,
       int32_t const& coord_q,
       int32_t const& coord_p,
       int32_t const& coord_z,
       int32_t const& coord_n,
       int8_t  const& weight_pos_s,
       int8_t  const& weight_pos_r,
       int8_t  const& weight_pos_t,
       int32_t const& range_c,
       int32_t const& range_nzpq,
       mute::v3i32_t const& conv_param,
       mute::v3i32_t const& output_dim)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v5i32_t crd {coord_c, coord_q, coord_p, coord_z, coord_n};
    mute::v2i32_t blk_dim{range_c, range_nzpq};
    detail::Im2ColWeightPos weight_pos{weight_pos_s, weight_pos_r, weight_pos_t};
    __musa_tme_ld_im2col_5d(bar_id, make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, blk_dim, crd,
                            static_cast<int32_t>(weight_pos), output_dim, conv_param, static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl),
                            static_cast<int32_t>(prefetch));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }

  struct PREFETCH
  {
    MUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& coord_c,
         int32_t const& coord_q,
         int32_t const& coord_p,
         int32_t const& coord_z,
         int32_t const& coord_n,
         int8_t  const& weight_pos_s,
         int8_t  const& weight_pos_r,
         int8_t  const& weight_pos_t,
         int32_t const& range_c,
         int32_t const& range_nzpq,
         mute::v3i32_t const& conv_param,
         mute::v3i32_t const& output_dim)
    {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      mute::v5i32_t crd {coord_c, coord_q, coord_p, coord_z, coord_n};
      mute::v2i32_t blk_dim{range_c, range_nzpq};
      detail::Im2ColWeightPos weight_pos{weight_pos_s, weight_pos_r, weight_pos_t};
      __musa_tme_im2col_prefetch_only_5d(gmem_int_desc, blk_dim, crd, static_cast<int32_t>(weight_pos), output_dim, conv_param);
#else
      MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
    }
  };
};


template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl,
  PrefetchSize          prefetch
>
struct MP31_TME_LOAD_IM2COL_ENTRY
{

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& coord_c,
       int32_t const& coord_q,
       int32_t const& coord_n,
       int8_t  const& weight_pos_s,
       int32_t const& range_c,
       int32_t const& range_nzpq,
       mute::v3i32_t const& conv_param,
       int32_t const& output_dim_q)
  {
    return MP31_TME_LOAD_IM2COL_3D<sg, ss, sl, prefetch>::copy(desc_ptr, bar_id, smem_ptr,
                                                               coord_c, coord_q, coord_n,
                                                               weight_pos_s,
                                                               range_c, range_nzpq,
                                                               conv_param,
                                                               output_dim_q);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& coord_c,
       int32_t const& coord_q,
       int32_t const& coord_p,
       int32_t const& coord_n,
       int8_t  const& weight_pos_s,
       int8_t  const& weight_pos_r,
       int32_t const& range_c,
       int32_t const& range_nzpq,
       mute::v3i32_t const& conv_param,
       mute::v2i32_t const& output_dim)
  {
    return MP31_TME_LOAD_IM2COL_4D<sg, ss, sl, prefetch>::copy(desc_ptr, bar_id, smem_ptr,
                                                               coord_c, coord_q, coord_p, coord_n,
                                                               weight_pos_s, weight_pos_r,
                                                               range_c, range_nzpq,
                                                               conv_param,
                                                               output_dim);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint32_t const& bar_id,
       void      * smem_ptr,
       int32_t const& coord_c,
       int32_t const& coord_q,
       int32_t const& coord_p,
       int32_t const& coord_z,
       int32_t const& coord_n,
       int8_t  const& weight_pos_s,
       int8_t  const& weight_pos_r,
       int8_t  const& weight_pos_t,
       int32_t const& range_c,
       int32_t const& range_nzpq,
       mute::v3i32_t conv_param,
       mute::v3i32_t output_dim)
  {
    return MP31_TME_LOAD_IM2COL_5D<sg, ss, sl, prefetch>::copy(desc_ptr, bar_id, smem_ptr,
                                                               coord_c, coord_q, coord_p, coord_z, coord_n,
                                                               weight_pos_s, weight_pos_r, weight_pos_t,
                                                               range_c, range_nzpq,
                                                               conv_param,
                                                               output_dim);
  }

};



////////////////////////////////////////////////////////////////////////////////////////////////////
/// TME_STORE : Initiates a TME store, in tile mode, from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP31_TME_STORE {};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl
>
struct MP31_TME_STORE_1D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& dim0)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    __musa_tme_st_1d(make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim0, crd0,
                     static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl
>
struct MP31_TME_STORE_2D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& dim0,
       int32_t const& dim1)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v2i32_t crd {crd0, crd1};
    mute::v2i32_t dim {dim0, dim1};
    __musa_tme_st_2d(make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim, crd,
                     static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl
>
struct MP31_TME_STORE_3D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v3i32_t crd {crd0, crd1, crd2};
    mute::v3i32_t dim {dim0, dim1, dim2};
    __musa_tme_st_3d(make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim, crd,
                     static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl
>
struct MP31_TME_STORE_4D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v4i32_t crd {crd0, crd1, crd2, crd3};
    mute::v4i32_t dim {dim0, dim1, dim2, dim3};
    __musa_tme_st_4d(make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim, crd,
                     static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl
>
struct MP31_TME_STORE_5D
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& crd4,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3,
       int32_t const& dim4)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    mute::v5i32_t crd {crd0, crd1, crd2, crd3, crd4};
    mute::v5i32_t dim {dim0, dim1, dim2, dim3, dim4};
    __musa_tme_st_5d(make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr), gmem_int_desc, dim, crd,
                     static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }
};

template <
  TME::SmemSwizzleGranularity sg,
  TME::SmemSwizzleStride      ss,
  TME::SmemSwizzleLine        sl
>
struct MP31_TME_STORE_ENTRY
{
  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& dim0)
  {
    return MP31_TME_STORE_1D<sg, ss, sl>::copy(desc_ptr, smem_ptr, crd0, dim0);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& dim0,
       int32_t const& dim1)
  {
    return MP31_TME_STORE_2D<sg, ss, sl>::copy(desc_ptr, smem_ptr, crd0, crd1, dim0, dim1);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2)
  {
    return MP31_TME_STORE_3D<sg, ss, sl>::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, dim0, dim1, dim2);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3)
  {
    return MP31_TME_STORE_4D<sg, ss, sl>::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3, dim0, dim1, dim2, dim3);
  }

  MUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0,
       int32_t const& crd1,
       int32_t const& crd2,
       int32_t const& crd3,
       int32_t const& crd4,
       int32_t const& dim0,
       int32_t const& dim1,
       int32_t const& dim2,
       int32_t const& dim3,
       int32_t const& dim4)
  {
    return MP31_TME_STORE_5D<sg, ss, sl>::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3, crd4, dim0, dim1, dim2, dim3, dim4);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// BLK_COPY : Copy a bulk of memory between shared memory and global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct MP31_BLK_COPY_G2S
{
  MUTE_HOST_DEVICE static void
  copy(void const* gmem_ptr, int32_t  const& bar_id,
       void      * smem_ptr, uint32_t const& load_bytes)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    constexpr TME::SmemSwizzleGranularity sg = TME::SmemSwizzleGranularity::NONE;
    constexpr TME::SmemSwizzleStride      ss = TME::SmemSwizzleStride::B256;
    constexpr TME::SmemSwizzleLine        sl = TME::SmemSwizzleLine::B256;
    constexpr PrefetchSize          prefetch = PrefetchSize::B128;

    uint64_t gmem_int_ptr = reinterpret_cast<uint64_t>(gmem_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    __musa_tme_ld_blk(make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
                      make_ptr_with_address_space<AddressSpace::Global>(gmem_int_ptr),
                      load_bytes, bar_id,
                      static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl), static_cast<int32_t>(prefetch));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }
};

struct MP31_BLK_COPY_S2G
{
  MUTE_HOST_DEVICE static void
  copy(void const* smem_ptr,
       void      * gmem_ptr, uint32_t const& store_bytes)
  {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
    constexpr TME::SmemSwizzleGranularity sg = TME::SmemSwizzleGranularity::NONE;
    constexpr TME::SmemSwizzleStride      ss = TME::SmemSwizzleStride::B256;
    constexpr TME::SmemSwizzleLine        sl = TME::SmemSwizzleLine::B256;

    uint64_t gmem_int_ptr = reinterpret_cast<uint64_t>(gmem_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    __musa_tme_st_blk(make_ptr_with_address_space<AddressSpace::Shared>(smem_int_ptr),
                      make_ptr_with_address_space<AddressSpace::Global>(gmem_int_ptr),
                      store_bytes,
                      static_cast<int32_t>(sg), static_cast<int32_t>(ss), static_cast<int32_t>(sl));
#else
    MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
  }
};

struct MP31_BLK_COPY_AUTO {};

// Indicate arrival of warp issuing TME_STORE
MUTE_HOST_DEVICE static void
tme_store_arrive() {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
  __musa_tme_store_commit();
#else
  MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
}

// Wait until at most Count committed TME_STOREs are pending and all prior commits are complete
MUTLASS_HOST_DEVICE static void
tme_store_wait() {
#if defined(MUTE_ARCH_TME_MP31_ACTIVATED)
  __musa_tme_store_read_wait();
#else
  MUTE_INVALID_CONTROL_PATH("Trying to use tme without MUTE_ARCH_TME_MP31_ACTIVATED.");
#endif
}

} // namespace mute
