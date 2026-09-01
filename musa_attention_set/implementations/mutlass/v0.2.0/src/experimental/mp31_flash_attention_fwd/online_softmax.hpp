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

 #include <mute/tensor.hpp>
 #include <mutlass/fast_math.h>
 
 using namespace mute;
 
 template <class LayoutAccum>
 MUTE_DEVICE
 auto
 convert_layout_acc_rowcol(LayoutAccum acc_layout) {
   // from ((2, VN, VM), MMA_M, MMA_N) to ((VM, MMA_M),(2, VN, MMA_N));
   static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
   static_assert(decltype(rank(acc_layout))::value == 3);
   auto l = acc_layout;
   return make_layout(make_layout(get<0,2>(l), get<1>(l)), make_layout(get<0,0>(l), get<0,1>(l), get<2>(l)));
 }
 
 
 template <typename T>
 struct MaxOp {
   MUTE_DEVICE T operator()(T const& x, T const& y) { return x > y ? x : y; }
 };
 
 template <>
 struct MaxOp<float> {
   MUTE_DEVICE float operator()(float const& x, float const& y) { return max(x, y); };
 };
 
 template <typename T>
 struct SumOp {
   MUTE_DEVICE T operator()(T const& x, T const& y) { return x + y; }
 };
 
 template <int THREADS>
 struct Allreduce {
   static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
   template <typename T, typename Operator>
   static
   MUTE_DEVICE T run(T x, Operator& op) {
     constexpr int OFFSET = THREADS / 2;
     x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
     return Allreduce<OFFSET>::run(x, op);
   }
 };
 
 template <>
 struct Allreduce<2> {
   template <typename T, typename Operator>
   static
   MUTE_DEVICE T run(T x, Operator& op) {
     x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
     return x;
   }
 };
 
 template <typename Tensor0, typename Tensor1, typename Operator>
 MUTE_DEVICE void octave_allreduce_(Tensor0& dst, Tensor1& src, Operator& op) {
   MUTE_STATIC_ASSERT_V(size(dst) == size(src));
 
   MUTE_UNROLL
   for (int i = 0; i < size(dst); ++i) {
     dst(i) = Allreduce<8>::run(src(i), op);
   }
 }
 
 template <int COLS>
 struct ThreadReduce {
   template <bool init, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
             typename Operator>
   MUTE_DEVICE void run(Tensor<Engine0, Layout0> const& tensor,
                        Tensor<Engine1, Layout1>& summary, Operator& op) {
     static_assert(Layout0::rank == 2, "Only support 2D Tensor");
     static_assert(Layout1::rank == 1, "Only support 1D Tensor");
     MUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
 
     MUTE_UNROLL
     for (int mi = 0; mi < size<0>(tensor); ++mi) {
       summary(mi) = init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
       MUTE_UNROLL
       for (int ni = 1; ni < size<1>(tensor); ++ni) {
         summary(mi) = op(summary(mi), tensor(mi, ni));
       }
     }
   }
 };
 
 template <>
 struct ThreadReduce<16> {
   template <bool init, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
             typename Operator>
   MUTE_DEVICE void run(Tensor<Engine0, Layout0> const& tensor,
                        Tensor<Engine1, Layout1>& summary, Operator& op) {
     static_assert(Layout0::rank == 2, "Only support 2D Tensor");
     static_assert(Layout1::rank == 1, "Only support 1D Tensor");
     MUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
 
     MUTE_UNROLL
     for (int mi = 0; mi < size<0>(tensor); ++mi) {
       // Manually unroll to trigger some nice optimizations
       Tensor reduce_group = make_tensor<typename Engine1::value_type>(Shape<_4>{});
 
       reduce_group(0) = op(init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0)), tensor(mi, 0x4));
       reduce_group(1) = op(tensor(mi, 0x1), tensor(mi, 0x5));
       reduce_group(2) = op(tensor(mi, 0x2), tensor(mi, 0x6));
       reduce_group(3) = op(tensor(mi, 0x3), tensor(mi, 0x7));
 
       reduce_group(0) = op(reduce_group(0), tensor(mi, 0x8));
       reduce_group(1) = op(reduce_group(1), tensor(mi, 0x9));
       reduce_group(2) = op(reduce_group(2), tensor(mi, 0xa));
       reduce_group(3) = op(reduce_group(3), tensor(mi, 0xb));
 
       reduce_group(0) = op(reduce_group(0), tensor(mi, 0xc));
       reduce_group(1) = op(reduce_group(1), tensor(mi, 0xd));
       reduce_group(2) = op(reduce_group(2), tensor(mi, 0xe));
       reduce_group(3) = op(reduce_group(3), tensor(mi, 0xf));
 
       reduce_group(0) = op(reduce_group(0), reduce_group(2));
       reduce_group(1) = op(reduce_group(1), reduce_group(3));
 
       summary(mi) = op(reduce_group(0), reduce_group(1));
     }
   }
 };
 
 //////////////////////////////////////////////////////////////////////////////////////////////////
 
 template <class Fragment0, class Fragment1, class AccumType>
 MUTE_HOST_DEVICE
 void
 online_softmax_scale(Fragment0& accum, Fragment1& accum_softmax, AccumType rln2_scale) {
   MUTE_UNROLL
   for (int i = 0; i < size(accum); ++i) {
     accum_softmax(i) = accum(i) * rln2_scale;
   }
 }

 template <class ElementAccum, class Fragment0, class Fragment1, class AccumType, class Fragment2>
 MUTE_HOST_DEVICE
 void
 online_softmax_scale_last(Fragment0& accum, Fragment1& accum_softmax, AccumType rln2_scale, Fragment2 tCcS, int oob_n) {

  MUTE_UNROLL
   for (int i = 0; i < size(accum_softmax); ++i) {
    if (get<1>(tCcS(i)) < oob_n) {
      accum_softmax(i) = accum(i) * rln2_scale;
    } else {
      accum_softmax(i) = -999999;
    }
   }
 }
 
 
 template <class ElementAccum, class Fragment0, class Fragmnet1>
 MUTE_HOST_DEVICE
 void
 online_softmax_row_max(Fragment0& accum_softmax,
                        Fragmnet1& row_max,
                        Fragmnet1& old_row_max,
                        Fragmnet1& alpha) {
   Tensor accum_s = make_tensor(accum_softmax.data(), convert_layout_acc_rowcol(accum_softmax.layout()));
 
   MUTE_STATIC_ASSERT_V(size(row_max) == size<0>(accum_s));
   MUTE_STATIC_ASSERT_V(size(old_row_max) == size<0>(accum_s));
   MUTE_STATIC_ASSERT_V(size(alpha) == size<0>(accum_s));
   //static_assert(size<1>(accum_s) == 16);

   MaxOp<ElementAccum> op;
   ThreadReduce<size<1>(accum_s)> thread_reduce;
   thread_reduce.template run<true>(accum_s, row_max, op);
   octave_allreduce_(row_max, row_max, op);
 
   MUTE_UNROLL
   for (int mi = 0; mi < size<0>(accum_s); ++mi) {
     row_max(mi) = mutlass::fast_max(row_max(mi), old_row_max(mi));
 
     alpha(mi) = old_row_max(mi) - row_max(mi);
     alpha(mi) = exp2f(alpha(mi));
     old_row_max(mi) = row_max(mi);
   }

   MUTE_UNROLL
   for (int mi = 0; mi < size<0>(accum_s); ++mi) {
     MUTE_UNROLL
     for (int ni = 0; ni < size<1>(accum_s); ++ni) {
       accum_s(mi, ni) = accum_s(mi, ni) - row_max(mi);
     }
   }
 }
 
 template <class Fragment0, class Fragment1>
 MUTE_HOST_DEVICE
 void
 update_accum(Fragment0& accum, Fragment1& alpha) {
   Tensor accum_o = make_tensor(accum.data(), convert_layout_acc_rowcol(accum.layout()));
   MUTE_STATIC_ASSERT_V(size(alpha) == size<0>(accum_o));
 
   MUTE_UNROLL
   for (int mi = 0; mi < size<0>(accum_o); ++mi) {
     MUTE_UNROLL
     for (int ni = 0; ni < size<1>(accum_o); ++ni) {
       accum_o(mi, ni) *= alpha(mi);
     }
   }
 }

 template <class Fragmnet>
 MUTE_HOST_DEVICE
 void
 online_softmax_exp2(Fragmnet& accum) {
   MUTE_UNROLL
   for (int i = 0; i < size(accum); ++i) {
     accum(i) = exp2f(accum(i));
   }
 }
 
 template <class ElementAccum, class Fragment0, class Fragment1>
 MUTE_HOST_DEVICE
 void
 online_softmax_row_sum(Fragment0& accum_softmax, Fragment1& row_sum,
                        Fragment1& lse, Fragment1& alpha) {
   Tensor accum_s = make_tensor(accum_softmax.data(), convert_layout_acc_rowcol(accum_softmax.layout()));
   //static_assert(size<1>(accum_s) == 16);
   MUTE_STATIC_ASSERT_V(size<0>(accum_s) == size(row_sum));
   SumOp<ElementAccum> op;
   ThreadReduce<size<1>(accum_s)> thread_reduce;
   thread_reduce.template run<true>(accum_s, row_sum, op);
   octave_allreduce_(row_sum, row_sum, op);
 
   MUTE_UNROLL
   for (int mi = 0; mi < size<0>(accum_s); ++mi) {
     lse(mi) = lse(mi) * alpha(mi) + row_sum(mi);
   }
 }
 
 template <class Fragment0, class Fragment1>
 MUTE_HOST_DEVICE
 void
 apply_softmax_normalizer(Fragment0& accum, Fragment1& lse) {
   Tensor accum_o = make_tensor(accum.data(), convert_layout_acc_rowcol(accum.layout()));
   MUTE_STATIC_ASSERT_V(size<0>(accum_o) == size(lse));
 
   MUTE_UNROLL
   for (int i = 0; i < size(lse); ++i) {
     lse(i) = 1.0f / lse(i);
   }
   MUTE_UNROLL
   for (int mi = 0; mi < size<0>(accum_o); ++mi) {
     MUTE_UNROLL
     for (int ni = 0; ni < size<1>(accum_o); ++ni) {
       accum_o(mi, ni) *= lse(mi);
     }
   }
 }

 template <class Fragment0, class Fragment1>
 MUTE_HOST_DEVICE
 void
 add_mask(Fragment0& accum, Fragment1& tCcMask, int m_idx, int n_idx, float margin) {
   #pragma unroll
   for (int i = 0; i < size(accum); ++i) {
     if ((m_idx + get<0>(tCcMask(i)))/(n_idx + get<1>(tCcMask(i))) < margin)
      accum(i) += -std::numeric_limits<float>::infinity();
   }
 }
