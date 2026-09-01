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

#include "mutlass/mutlass.h"
#include "mutlass/gemm/gemm.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/epilogue/dispatch_policy.hpp"

#include "mute/tensor.hpp"
#include "mute/numeric/numeric_types.hpp"
#include "mute/util/type_traits.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace epilogue {
namespace collective {

namespace detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Stride>
constexpr bool
is_m_major() {
  return mutlass::gemm::detail::is_major<0,Stride>();
}

template <class Stride>
constexpr bool
is_n_major() {
  return mutlass::gemm::detail::is_major<1,Stride>();
}

template <class Stride>
constexpr bool
is_im2col() {
  return mute::is_same_v<Stride, mutlass::detail::TagToStrideC_t<mutlass::layout::TensorNWC>>
      || mute::is_same_v<Stride, mutlass::detail::TagToStrideC_t<mutlass::layout::TensorNHWC>>
      || mute::is_same_v<Stride, mutlass::detail::TagToStrideC_t<mutlass::layout::TensorNDHWC>>;
}

using mutlass::atomic_maximum;

template <class T>
static constexpr int elements_per_access_v = mutlass::sizeof_bits<uint32_t>::value / mutlass::sizeof_bits<T>::value;

template <class GmemLayoutTag>
static constexpr bool is_im2col_mode =
  mute::is_same_v<GmemLayoutTag, mutlass::layout::TensorNWC> ||
  mute::is_same_v<GmemLayoutTag, mutlass::layout::TensorNHWC> ||
  mute::is_same_v<GmemLayoutTag, mutlass::layout::TensorNDHWC>;

template <class T>
struct EmptyStorage {
  MUTLASS_HOST_DEVICE
  T* data() { return nullptr; }
};

template<class EpilogueSchedule, class Stride>
MUTLASS_HOST_DEVICE
auto get_epilogue_stride(Stride stride){
  if constexpr (mute::is_base_of_v<mutlass::gemm::EpilogueTransposed, EpilogueSchedule>) {
    return mute::make_stride(mute::get<1>(stride), mute::get<0>(stride), mute::get<2>(stride));
  }
  else {
    return stride;
  }
}

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithBias { 
  static constexpr bool value = false; 
  using type = typename ThreadEpilogueOp::ElementCompute; 
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithBias <ThreadEpilogueOp, mute::void_t<typename ThreadEpilogueOp::ElementBias>> { 
  static constexpr bool value = true; 
  using type = typename ThreadEpilogueOp::ElementBias; 
};
template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithPerChannelScaling {
  static constexpr bool value = false;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithPerChannelScaling <ThreadEpilogueOp, mute::enable_if_t<ThreadEpilogueOp::IsPerChannelScalingSupported>> {
  static constexpr bool value = true;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithActivation {
  static constexpr bool value = false;
  using type = void;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithActivation <ThreadEpilogueOp, mute::enable_if_t<ThreadEpilogueOp::IsEltActSupported>> {
  static constexpr bool value = true;
  using type = typename ThreadEpilogueOp::ActivationFn;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithElementwiseArguments : mute::false_type {};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithElementwiseArguments<
        ThreadEpilogueOp,
        mute::void_t<typename ThreadEpilogueOp::ElementwiseOp::Arguments>> : mute::true_type {};


} // namespace detail
} // namespace collective
} // namespace epilogue
} // namespace mutlass
