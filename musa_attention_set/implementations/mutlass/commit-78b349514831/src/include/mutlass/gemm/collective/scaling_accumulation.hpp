/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mute/algorithm/clear.hpp"
#include "mute/tensor.hpp"

namespace mutlass::gemm::collective {

template <
  class EngineAccum,
  class LayoutAccum>
struct ScalingAccumulation {
  using TensorAccum = mute::Tensor<EngineAccum, LayoutAccum>;
  using ElementAccumulator = typename EngineAccum::value_type;

  static_assert(is_static<LayoutAccum>::value, "Accumulator Layout should be static");
  static_assert(is_rmem<TensorAccum>::value, "Accumulator tensor must be rmem resident.");

private:
  TensorAccum& accum_;
  TensorAccum accum_temp_;

  uint32_t accum_promotion_interval_;         // defines the max num of executed MMAs after which accum should be promoted.
  uint32_t mma_count_per_mainloop_iteration_; // num of MMAs per k_tile of mainloop
  uint32_t mma_count_;                        // current executed MMAs
  uint32_t reset_accum_flag_;                 // accum needs to be zeroed or not.

  template <
    class EngineScale,
    class LayoutScale>
  MUTLASS_DEVICE
  void scale_core(const mute::Tensor<EngineScale, LayoutScale> &scale) {
    using TensorScale = mute::Tensor<EngineScale, LayoutScale>;

    static_assert(is_static<LayoutScale>::value, "Scale Layout should be static");
    static_assert(is_rmem<TensorScale>::value , "Scale tensor must be rmem resident.");

    static_assert(LayoutAccum{}.shape() == LayoutScale{}.shape(), "Accumulator and scale must have same shape.");

    warpsquad_wait();
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) += accum_temp_(i) * scale(i);
    }
  }

  template <
    class EngineScaleA,
    class LayoutScaleA,
    class EngineScaleB,
    class LayoutScaleB>
  MUTLASS_DEVICE
  void scale_core(const mute::Tensor<EngineScaleA, LayoutScaleA> &scaleA,
                  const mute::Tensor<EngineScaleB, LayoutScaleB> &scaleB) {
    using TensorScaleA = mute::Tensor<EngineScaleA, LayoutScaleA>;
    using TensorScaleB = mute::Tensor<EngineScaleB, LayoutScaleB>;

    static_assert(is_static<LayoutScaleA>::value, "ScaleA Layout should be static");
    static_assert(is_static<LayoutScaleB>::value, "ScaleB Layout should be static");
    static_assert(is_rmem<TensorScaleA>::value , "ScaleA tensor must be rmem resident.");
    static_assert(is_rmem<TensorScaleB>::value , "ScaleB tensor must be rmem resident.");

    static_assert(LayoutAccum{}.shape() == LayoutScaleA{}.shape(), "Accumulator and scaleA must have same shape.");
    static_assert(LayoutAccum{}.shape() == LayoutScaleB{}.shape(), "Accumulator and scaleB must have same shape.");

    warpsquad_wait();
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) += accum_temp_(i) * scaleA(i) * scaleB(i);
    }
  }


public:
  MUTLASS_DEVICE
  ScalingAccumulation(
      TensorAccum &accum,
      uint32_t accum_promotion_interval,
      uint32_t mma_count_per_mainloop_iteration)
      : accum_(accum),
        accum_promotion_interval_(accum_promotion_interval),
        mma_count_per_mainloop_iteration_(mma_count_per_mainloop_iteration),
        mma_count_(0),
        reset_accum_flag_(0)
  {
    accum_temp_ = mute::make_fragment_like(accum);
  }

  //
  // Methods (Common)
  //
  MUTLASS_DEVICE
  TensorAccum& operator()() {
    return accum_temp_;
  }

  MUTLASS_DEVICE
  bool prepare_if_needed() {
    return reset_accum_flag_;
  }

  /// scale (multiply_add) the results from the MMA accumulators to main accumulator if needed.

  template <class ElementScale>
  MUTLASS_DEVICE
  void scale_if_needed(ElementScale const &scale) {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = mma_count_ == accum_promotion_interval_;
    if (reset_accum_flag_) {
      scale_core(scale);
      mma_count_ = 0;
    }
  }

  template <
    class EngineScale,
    class LayoutScale>
  MUTLASS_DEVICE
  void scale_if_needed(const mute::Tensor<EngineScale, LayoutScale> &scale) {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = mma_count_ == accum_promotion_interval_;
    if (reset_accum_flag_) {
      scale_core(scale);
      mma_count_ = 0;
    }
  }

  template <
    class EngineScaleA,
    class LayoutScaleA,
    class EngineScaleB,
    class LayoutScaleB>
  MUTLASS_DEVICE
  void scale_if_needed(const mute::Tensor<EngineScaleA, LayoutScaleA> &scaleA,
                       const mute::Tensor<EngineScaleB, LayoutScaleB> &scaleB) {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = mma_count_ == accum_promotion_interval_;
    if (reset_accum_flag_) {
      scale_core(scaleA, scaleB);
      mma_count_ = 0;
    }
  }



  /// scale (multiply_add) the residue results from the MMA accumulators to main accumulator if needed.
  template <class ElementScale>
  MUTLASS_DEVICE
  void scale_residue_if_needed(ElementScale const &scale) {
    if (mma_count_ > 0) {
      scale_core(scale);
    }
  }

  template <
    class EngineScale,
    class LayoutScale>
  MUTLASS_DEVICE
  void scale_residue_if_needed(const mute::Tensor<EngineScale, LayoutScale> &scale) {
    if (mma_count_ > 0) {
      scale_core(scale);
    }
  }

  template <
    class EngineScaleA,
    class LayoutScaleA,
    class EngineScaleB,
    class LayoutScaleB>
  MUTLASS_DEVICE
  void scale_residue_if_needed(const mute::Tensor<EngineScaleA, LayoutScaleA> &scaleA,
                               const mute::Tensor<EngineScaleB, LayoutScaleB> &scaleB) {
    if (mma_count_ > 0) {
      scale_core(scaleA, scaleB);
    }
  }
};

// Struct for iteration-scale. 
template <
  class EngineAccum, class LayoutAccum,
  class EngineScaleA_, class LayoutScaleA_,
  class EngineScaleB_, class LayoutScaleB_>
struct ScalingAccumulation_Iterative {
  using TensorAccum = mute::Tensor<EngineAccum, LayoutAccum>;
  using TensorScaleA = mute::Tensor<EngineScaleA_, LayoutScaleA_>;
  using TensorScaleB = mute::Tensor<EngineScaleB_, LayoutScaleB_>;
  using ElementAccumulator = typename EngineAccum::value_type;
  using ElementScaleBlock = typename EngineScaleA_::value_type;

  static_assert(is_static<LayoutAccum>::value, "Accumulator Layout should be static");
  static_assert(is_rmem<TensorAccum>::value, "Accumulator tensor must be rmem resident.");

private:
  TensorAccum& accum_;
  TensorScaleA iterative_scaleA;
  TensorScaleB iterative_scaleB;

  uint32_t accum_promotion_interval_;         // defines the max num of executed MMAs after which accum should be promoted.
  uint32_t mma_count_per_mainloop_iteration_; // num of MMAs per k_tile of mainloop
  uint32_t mma_count_;                        // current executed MMAs
  uint32_t reset_accum_flag_;                 // accum needs to be zeroed or not.

  template <
    class EngineScale,
    class LayoutScale>
  MUTLASS_DEVICE
  void scale_core(const mute::Tensor<EngineScale, LayoutScale> &scale) {
    using TensorScale = mute::Tensor<EngineScale, LayoutScale>;

    static_assert(is_static<LayoutScale>::value, "Scale Layout should be static");
    static_assert(is_rmem<TensorScale>::value , "Scale tensor must be rmem resident.");

    static_assert(LayoutAccum{}.shape() == LayoutScale{}.shape(), "Accumulator and scale must have same shape.");

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) *= scale(i);
    }
  }

  template <
    class EngineScaleA,
    class LayoutScaleA,
    class EngineScaleB,
    class LayoutScaleB>
  MUTLASS_DEVICE
  void scale_core(const mute::Tensor<EngineScaleA, LayoutScaleA> &scaleA,
                  const mute::Tensor<EngineScaleB, LayoutScaleB> &scaleB) {
    using TensorScaleA = mute::Tensor<EngineScaleA, LayoutScaleA>;
    using TensorScaleB = mute::Tensor<EngineScaleB, LayoutScaleB>;

    static_assert(is_static<LayoutScaleA>::value, "ScaleA Layout should be static");
    static_assert(is_static<LayoutScaleB>::value, "ScaleB Layout should be static");
    static_assert(is_rmem<TensorScaleA>::value , "ScaleA tensor must be rmem resident.");
    static_assert(is_rmem<TensorScaleB>::value , "ScaleB tensor must be rmem resident.");

    static_assert(LayoutAccum{}.shape() == LayoutScaleA{}.shape(), "Accumulator and scaleA must have same shape.");
    static_assert(LayoutAccum{}.shape() == LayoutScaleB{}.shape(), "Accumulator and scaleB must have same shape.");

    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) *= scaleA(i) * scaleB(i);
    }
  }

  MUTLASS_DEVICE
  void div_coreA() {
    //warpsquad_wait();
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) *= iterative_scaleA(i);
    }
  }

  MUTLASS_DEVICE
  void div_coreB() {
    //warpsquad_wait();
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) *= iterative_scaleB(i);
    }
  }

  MUTLASS_DEVICE
  void div_coreAB() {
    //warpsquad_wait();
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) *= iterative_scaleA(i) * iterative_scaleB(i);
      //accum_(i) *= iterative_scaleB(i);
    }
  }


public:
  MUTLASS_DEVICE
  ScalingAccumulation_Iterative(
      TensorAccum &accum,
      uint32_t accum_promotion_interval,
      uint32_t mma_count_per_mainloop_iteration,
      const TensorScaleA &scaleA,
      const TensorScaleB &scaleB
      )
      : accum_(accum),
        accum_promotion_interval_(accum_promotion_interval),
        mma_count_per_mainloop_iteration_(mma_count_per_mainloop_iteration),
        mma_count_(0),
        reset_accum_flag_(0) {
          iterative_scaleA = mute::make_tensor_like<ElementScaleBlock>(scaleA);
          iterative_scaleB = mute::make_tensor_like<ElementScaleBlock>(scaleB);
        }

  //
  // Methods (Common)
  //
  MUTLASS_DEVICE
  TensorAccum& operator()() {
    return accum_;
  }

  MUTLASS_DEVICE
  bool prepare_if_needed() {
    //return reset_accum_flag_;
    return mma_count_ == 0;
  }

  /// scale (multiply_add) the results from the MMA accumulators to main accumulator if needed.

  MUTLASS_DEVICE
  void advance() {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = mma_count_ == accum_promotion_interval_;
    if (reset_accum_flag_) {
      mma_count_ = 0;
    }
  }


  /// scale (multiply_add) the residue results from the MMA accumulators to main accumulator if needed.
  template <class ElementScale>
  MUTLASS_DEVICE
  void scale_residue_if_needed(ElementScale const &scale) {
    scale_core(scale);
  }

  template <
    class EngineScale,
    class LayoutScale>
  MUTLASS_DEVICE
  void scale_residue_if_needed(const mute::Tensor<EngineScale, LayoutScale> &scale) {
    scale_core(scale);
  }
  
  template <
    class EngineScaleA,
    class LayoutScaleA,
    class EngineScaleB,
    class LayoutScaleB>
  MUTLASS_DEVICE
  void scale_residue_if_needed(const mute::Tensor<EngineScaleA, LayoutScaleA> &scaleA,
                               const mute::Tensor<EngineScaleB, LayoutScaleB> &scaleB) {
    scale_core(scaleA, scaleB);
  }

  MUTLASS_DEVICE
  void div_if_needA(){
    div_coreA();
  }
  
  MUTLASS_DEVICE
  void div_if_needB(){
    div_coreB();
  }

  MUTLASS_DEVICE
  void div_if_needAB(){
    div_coreAB();
  }

  template <
    class GTensor,
    class RTensor>
  MUTLASS_DEVICE
  void copyA(GTensor &gscaleA, RTensor &rscaleA){
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(mute::filter_zeros(rscaleA)); i++)
      mute::filter_zeros(iterative_scaleA)(i) = mute::filter_zeros(rscaleA)(i);
    mute::copy(gscaleA, rscaleA);
    //MUTLASS_PRAGMA_UNROLL
    //for (int i = 0; i < size(mute::filter_zeros(rscaleA)); i++){
    //  mute::filter_zeros(rscaleA)(i) += ElementScaleBlock(1e-16);
    //}
  }
  template <
    class GTensor,
    class RTensor>
  MUTLASS_DEVICE
  void copyB(GTensor &gscaleB, RTensor &rscaleB){
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(mute::filter_zeros(rscaleB)); i++)
      mute::filter_zeros(iterative_scaleB)(i) = mute::filter_zeros(rscaleB)(i);
    mute::copy(gscaleB, rscaleB);
    //MUTLASS_PRAGMA_UNROLL
    //for (int i = 0; i < size(mute::filter_zeros(rscaleB)); i++){
    //  mute::filter_zeros(rscaleB)(i) += ElementScaleBlock(1e-16);
    //}
  }

  template <
    class TensorA>
  MUTLASS_DEVICE
  void update_iterationA(TensorA &rscaleA){
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(mute::filter_zeros(rscaleA)); i++){
      mute::filter_zeros(iterative_scaleA)(i) *= __frcp_rn(mute::filter_zeros(rscaleA)(i));
    }
  }

  template <
    class TensorB>
  MUTLASS_DEVICE
  void update_iterationB(TensorB &rscaleB){
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(mute::filter_zeros(rscaleB)); i++){
      mute::filter_zeros(iterative_scaleB)(i) *= __frcp_rn(mute::filter_zeros(rscaleB)(i));
    }
  }

  template <
    class TensorA,
    class TensorB>
  MUTLASS_DEVICE
  void update_iterationAB( TensorA &rscaleA,
                        TensorB &rscaleB){
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(mute::filter_zeros(iterative_scaleA)); i++){
      mute::filter_zeros(iterative_scaleA)(i) *= __frcp_rn(mute::filter_zeros(rscaleA)(i));
    }
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(mute::filter_zeros(iterative_scaleB)); i++){
      mute::filter_zeros(iterative_scaleB)(i) *= __frcp_rn(mute::filter_zeros(rscaleB)(i));
    }
  }
};
} // namespace mutlass::gemm::collective
