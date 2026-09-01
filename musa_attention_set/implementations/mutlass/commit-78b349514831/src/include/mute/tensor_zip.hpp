/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2024 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <mute/config.hpp>           // MUTE_HOST_DEVICE
#include <mute/tensor_impl.hpp>      // mute::Tensor
#include <mute/container/tuple.hpp>  // mute::tuple

namespace mute
{

// A tuple of Iterators that can be offset asymmetrically
// Note that this only accepts op+(tuple<Index...>) and op[tuple<Index...>]
//   where each iterator will be offset by its respective index only.
// READ-ONLY for now until mute::tuple can be constructed with references.
template <class... Iters>
struct ZipIterator
{
  using value_type   = mute::tuple<iter_value_t<Iters>...>;
  using element_type = mute::tuple<iter_element_t<Iters>...>;
  // NOTE: mute::tuple does not support constructions with references at the moment.
  //       Consider fixes and/or an implementation of std::forward_as_tuple.
  //       For now, use a mute::tuple of value_types instead, which makes this Iterator READ-ONLY.
  //using reference    = mute::tuple<iter_reference_t<Iters>...>;
  using reference  = value_type;

  ZipIterator() = delete;

  MUTE_HOST_DEVICE constexpr
  ZipIterator(Iters... iters)
    : iters_(iters...)
  {}

  MUTE_HOST_DEVICE constexpr
  ZipIterator(mute::tuple<Iters...> const& iters)
    : iters_(iters)
  {}

  MUTE_HOST_DEVICE constexpr
  reference operator*() const {
    return mute::apply(iters_, [](auto&&... args) { return reference(*args...); });
  }

  template <class... Index>
  MUTE_HOST_DEVICE constexpr
  ZipIterator operator+(mute::tuple<Index...> const& idxs) const {
    static_assert(sizeof...(Index) == sizeof...(Iters), "Expect same number of offsets as iterators.");
    return mute::transform(iters_, idxs, [](auto&& iter, auto&& idx) { return iter + idx; });
  }

  template <class... Index>
  MUTE_HOST_DEVICE constexpr
  reference operator[](mute::tuple<Index...> const& idxs) const {
    return *(*this + idxs);
  }

  mute::tuple<Iters...> iters_;
};

//------------------------------------------------------------------------------
// type traits

template <class... Iters>
struct is_rmem<ZipIterator<Iters...>> : conjunction<is_rmem<Iters>...> {};
template <class... Iters>
struct is_smem<ZipIterator<Iters...>> : conjunction<is_smem<Iters>...> {};
template <class... Iters>
struct is_gmem<ZipIterator<Iters...>> : conjunction<is_gmem<Iters>...> {};
// A tuple of Layouts that operates on each Layout symmetrically
// The Layouts need to have compatible shapes and ranks.
// The ZipLayout presents the intersection of the domain of its component Layouts.
//   E.g. all Layouts accept 1D coords and ZipLayout does as well.
// The ZipLayout returns the union of the codomain of its component Layouts.
//   E.g. all Layouts return an integer so ZipLayout returns a tuple of integers.
template <class... Layouts>
struct ZipLayout
{
  static constexpr int rank = (int(0) | ... | Layouts::rank);

  static_assert((is_layout<Layouts>::value && ...), "All template parameters must be layouts");
  static_assert(((Layouts::rank == rank) && ...),   "All layouts must have the same rank");

  MUTE_HOST_DEVICE constexpr
  ZipLayout(Layouts const&... layouts)
    : layouts_(layouts...)
  {}

  MUTE_HOST_DEVICE constexpr
  ZipLayout(mute::tuple<Layouts...> const& layouts)
    : layouts_(layouts)
  {}

  template <class Coord>
  MUTE_HOST_DEVICE constexpr
  auto
  operator()(Coord const& coord) const {
    if constexpr (has_underscore<Coord>::value) {
      return ZipLayout(mute::transform(layouts_, [&] (auto layout) { return layout(coord); }));
    } else {
      return mute::transform(layouts_, [&] (auto layout) { return layout(coord); });
    }

    MUTE_GCC_UNREACHABLE;
  }

  // op() convenience function for multi-dimensional coordinates
  template <class Coord0, class Coord1, class... Coords>
  MUTE_HOST_DEVICE constexpr
  decltype(auto)
  operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const {
    return operator()(make_coord(c0,c1,cs...));
  }

  mute::tuple<Layouts...> layouts_;
};

template <class... Layouts>
struct is_layout<ZipLayout<Layouts...>> : true_type {};

//
// make_zip_tensor and unzip_tensor
//

template <class... Engines, class... Layouts>
MUTE_HOST_DEVICE constexpr
auto
make_zip_tensor(Tensor<Engines,Layouts> const&... tensors)
{
  return make_tensor(ZipIterator(tensors.data()...),
                     ZipLayout(tensors.layout()...));
}

template <class Engine, class Layout>
MUTE_HOST_DEVICE constexpr
auto
unzip_tensor(Tensor<Engine,Layout> const& tensor)
{
  return mute::transform(tensor.data().iters_, tensor.layout().layouts_,
                         [](auto iter, auto layout) { return make_tensor(iter, layout); });
}

//
// Utilities
//

template <int... Is, class... Layouts>
MUTE_HOST_DEVICE constexpr
auto
rank(ZipLayout<Layouts...> const& layouts)
{
  return rank<Is...>(get<0>(layouts.layouts_));
}

template <int... Is, class... Layouts>
MUTE_HOST_DEVICE constexpr
auto
size(ZipLayout<Layouts...> const& layouts)
{
  return size<Is...>(get<0>(layouts.layouts_));
}

//
// Manipulation
//

// Extend each component layout to rank-N by appending Layout @a x.
template <int N, class... Layouts, class ShapeX = _1, class StrideX = _0>
MUTE_HOST_DEVICE constexpr
auto
append(ZipLayout<Layouts...>  const& layouts,
       Layout<ShapeX,StrideX> const& x = {})
{
  return ZipLayout(mute::transform(layouts.layouts_, [&](auto t){ return append<N>(t, x); }));
}

// Extend each component layout to rank-N by prepending Layout @a x.
template <int N, class... Layouts, class ShapeX = _1, class StrideX = _0>
MUTE_HOST_DEVICE constexpr
auto
prepend(ZipLayout<Layouts...>  const& layouts,
        Layout<ShapeX,StrideX> const& x = {})
{
  return ZipLayout(mute::transform(layouts.layouts_, [&](auto t){ return prepend<N>(t, x); }));
}

template <class... Layouts, class Tiler>
MUTE_HOST_DEVICE constexpr
auto
logical_divide(ZipLayout<Layouts...> const& layouts,
               Tiler                 const& tiler)
{
  return ZipLayout(mute::transform(layouts.layouts_, [&](auto t){ return logical_divide(t, tiler); }));
}

template <class... Layouts, class Tiler>
MUTE_HOST_DEVICE constexpr
auto
zipped_divide(ZipLayout<Layouts...> const& layouts,
              Tiler                 const& tiler)
{
  return ZipLayout(mute::transform(layouts.layouts_, [&](auto t){ return zipped_divide(t, tiler); }));
}

// Return <SlicedZipLayout, ZipOffsets> by calling slice_and_offset and all component layouts.
template <class Coord, class... Layouts>
MUTE_HOST_DEVICE constexpr
auto
slice_and_offset(Coord const& c, ZipLayout<Layouts...> const& layouts)
{
  auto result = mute::zip(mute::transform(layouts.layouts_, [&c](auto const& layout) { return slice_and_offset(c, layout); }));
  return mute::make_tuple(ZipLayout(get<0>(result)), get<1>(result));
}

} // end namespace mute