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

/***************************************************************************************************
 * Copyright (c) 2017 - 2023 COLFAX
 * Distributed under MIT License
 **************************************************************************************************/

#pragma once

#include <vector>

#include "mutlass/util/device_memory.h"
#include "mutlass/util/reference/device/gemm_complex.h"
#include "mutlass/util/reference/host/gemm_complex.h"
#include "mutlass/util/tensor_view_io.h"

#define DEVICE_REF 0

/* https://github.com/ColfaxResearch/cutlass-kernels/blob/master/include/utils/random.hpp */
template <typename Element>
bool verify_tensor(std::vector<Element> vector_Input,
                   std::vector<Element> vector_Input_Ref,
                   bool printValues = false, bool printDiffs = false,
                   float errCountExpected = 0, int64_t verify_length = -1) {
  int64_t size = (vector_Input.size() < vector_Input_Ref.size())
      ? vector_Input.size()
      : vector_Input_Ref.size();
  size = (verify_length == -1) ? size : verify_length;

  float abs_tol = 1e-3f;

  float rel_tol = 1e-3f;
  int errCount = 0;
  for (int64_t i = 0; i < size; ++i) {
    if (printValues)
      std::cout << vector_Input[i] << " " << vector_Input_Ref[i] << std::endl;
    float diff = (float)(vector_Input[i] - vector_Input_Ref[i]);
    float abs_diff = fabs(diff);
    float abs_ref = fabs((float)vector_Input_Ref[i] + 1e-5f);
    float relative_diff = abs_diff / abs_ref;
    if ((isnan(vector_Input_Ref[i]) || isnan(abs_diff) || isinf(abs_diff)) ||
        (abs_diff > abs_tol && relative_diff > rel_tol)) {
      if (printDiffs)
        printf("[%d/%d] diff = %f, rel_diff = %f, {computed=%f, ref=%f}.\n",
               int(i), int(size), abs_diff, relative_diff,
               (float)(vector_Input[i]), (float)(vector_Input_Ref[i]));
      errCount++;
      return false;
    }
  }
  auto errCountComputed = float(errCount) / float(size) * 100;

  return errCountComputed <= errCountExpected ? true : false;
}

/* https://github.com/ColfaxResearch/cutlass-kernels/blob/master/include/utils/fmha_cutlass.hpp */
template <typename Element, typename AccumType>
class TestAttention {
public:
  //
  // Type definitions
  //
  static constexpr float kLog2e = float(1.4426950408889634074); // log_2(e) = M_LOG2E
  using ElementQ = Element;
  using ElementK = Element;
  using ElementS = Element;
  using ElementP = AccumType;
  using ElementAccumulator = AccumType;
  using ElementV = Element;
  using ElementO = Element;

  using ElementCompute = AccumType;

  using ElementNorm = AccumType;
  using ElementSum = AccumType;
  using ElementSoftmaxCompute = AccumType;

  using LayoutQ = mutlass::layout::RowMajor;
  using LayoutK = mutlass::layout::ColumnMajor;
  using LayoutP = mutlass::layout::RowMajor;
  using LayoutV = mutlass::layout::RowMajor;
  using LayoutO = mutlass::layout::RowMajor;
  using LayoutNorm = mutlass::layout::RowMajor;

  using MatrixCoord = typename LayoutP::TensorCoord;

private:
  bool help;
  bool error;
  bool reference_check;
  bool use_mask;
  bool causal;

  int alignment;
  int iterations;

  int QSeqLen;
  int HeadDimQK;
  int HeadDimV;
  int KVSeqLen;
  int HeadNumQ;
  int HeadNumKV;
  int BatchSize;

  // alpha0, alpha1 and beta are fixed
  // in this multi-head attention example
  float alpha0;
  float alpha1;
  float beta;

  mutlass::gemm::GemmCoord problem0;
  mutlass::gemm::GemmCoord problem1;
  //

  int64_t ldq;
  int64_t ldk;
  int64_t ldp;
  int64_t ldv;
  int64_t ldo;

  int64_t elements_Q;
  int64_t elements_K;
  int64_t elements_V;
  int64_t elements_P;
  int64_t elements_O;
  int64_t elements_norm;

public:
  //
  // Methods
  //

  TestAttention(int _head_number_q, int _head_number_kv, int _batch_size, int _head_dim_qk, int _head_dim_v,
                int _seq_length_q, int _seq_length_kv, int _alignment = 1, bool use_mask = false,
                bool causal = false) {

    QSeqLen = _seq_length_q;
    HeadDimQK = _head_dim_qk;
    HeadDimV = _head_dim_v;
    KVSeqLen = _seq_length_kv;
    BatchSize = _batch_size;
    HeadNumQ = _head_number_q;
    HeadNumKV = _head_number_kv;

    mutlass::gemm::GemmCoord _problem0(QSeqLen, KVSeqLen, HeadDimQK);
    mutlass::gemm::GemmCoord _problem1(QSeqLen, HeadDimV, KVSeqLen);

    problem0 = _problem0;
    problem1 = _problem1;
  }

public:
  /// Initializes data structures
  void initialize() {

    //
    // Set scaling factor(s).
    //
    alpha0 = 1.0f / sqrt(float(HeadDimQK));
    alpha1 = 1.0f;
    beta = 0;

    elements_Q = problem0.m() * problem0.k();
    elements_K = problem0.n() * problem0.k();
    elements_P = problem0.m() * problem0.n();
    elements_V = problem1.n() * problem1.k();
    elements_O = problem1.m() * problem1.n();
    elements_norm = problem0.m();

    ldq = LayoutQ::packed({problem0.m(), problem0.k()}).stride(0);
    ldk = LayoutK::packed({problem0.k(), problem0.n()}).stride(0);
    ldp = LayoutP::packed({problem0.m(), problem0.n()}).stride(0);
    ldv = LayoutV::packed({problem1.k(), problem1.n()}).stride(0);
    ldo = LayoutO::packed({problem1.m(), problem1.n()}).stride(0);
  }


  void compute(const ElementQ *Q, const ElementK *K, const ElementV *V,
               ElementS *S, ElementO *O, ElementNorm *norm, ElementSum *sum,
               bool usePow2 = false, bool usePreScaling = true, bool with_softmax = false) {
    LayoutQ layout_Q(ldq);
    LayoutK layout_K(ldk);
    LayoutP layout_P(ldp);
    LayoutV layout_V(ldv);
    LayoutO layout_O(ldo);

    LayoutNorm layout_norm(1);

    MatrixCoord extent_Q{problem0.m(), problem0.k()};
    MatrixCoord extent_K{problem0.k(), problem0.n()};
    MatrixCoord extent_P{problem0.m(), problem0.n()};
    MatrixCoord extent_V{problem1.k(), problem1.n()};
    MatrixCoord extent_O{problem1.m(), problem1.n()};
    MatrixCoord extent_norm{problem1.m(), 1};


#if DEVICE_REF
    mutlass::DeviceAllocation<ElementP> softmaxP(layout_P.capacity(extent_P));
    mutlass::TensorView<ElementP, LayoutP> softmaxViewP(softmaxP.get(),
                                                        layout_P, extent_P);
#else
    std::vector<ElementP> softmaxP(layout_P.capacity(extent_P));
    mutlass::TensorView<ElementP, LayoutP> softmaxViewP(softmaxP.data(),
                                                        layout_P, extent_P);

#endif

    for (int64_t b = 0; b < BatchSize; ++b) {
      for (int64_t h = 0; h < HeadNumQ; ++h) {
        int hkv = h%HeadNumKV;
        auto offsetQ = Q + b * HeadNumQ * elements_Q + h * elements_Q;
        auto offsetK = K + b * HeadNumKV * elements_K + hkv * elements_K;
        auto offsetS = S + b * HeadNumQ * elements_P + h * elements_P;
        auto offsetV = V + b * HeadNumKV * elements_V + hkv * elements_V;
        auto offsetO = O + b * HeadNumQ * elements_O + h * elements_O;

        auto offsetNorm = norm + b * HeadNumQ * elements_norm + h * elements_norm;
        auto offsetSum  =  sum + b * HeadNumQ * elements_norm + h * elements_norm;
        mutlass::TensorView<ElementQ, LayoutQ> view_Q(
            const_cast<ElementQ *>(offsetQ), layout_Q, extent_Q);
        mutlass::TensorView<ElementK, LayoutK> view_K(
            const_cast<ElementK *>(offsetK), layout_K, extent_K);
        mutlass::TensorView<ElementV, LayoutV> view_V(
            const_cast<ElementK *>(offsetV), layout_V, extent_V);

        mutlass::TensorView<ElementS, LayoutP> view_S(offsetS, layout_P, extent_P);
        mutlass::TensorView<ElementO, LayoutO> view_O(offsetO, layout_O,
                                                      extent_O);

        mutlass::TensorView<ElementNorm, LayoutNorm> view_Norm_Ref(
            offsetNorm, layout_norm, extent_norm);
        mutlass::TensorView<ElementSum, LayoutNorm> view_Sum_Ref(
            offsetSum, layout_norm, extent_norm);


        // Reference GEMM-I.

        float alphaMma0 = usePreScaling ? alpha0 : 1.0f;
        float postScaling = usePreScaling ? 1.0f : alpha0;

#if DEVICE_REF
        mutlass::DeviceAllocation<ElementP> block_Ref_P(
            layout_P.capacity(extent_P));
        mutlass::TensorView<ElementP, LayoutP> view_P(block_Ref_P.get(),
                                                      layout_P, extent_P);

        mutlass::reference::device::GemmComplex<
            ElementQ, LayoutQ, ElementK, LayoutK, ElementP, LayoutP,
            ElementAccumulator, ElementCompute>(
            problem0, ElementAccumulator(alphaMma0), view_Q,
            mutlass::ComplexTransform::kNone, view_K,
            mutlass::ComplexTransform::kNone, ElementAccumulator(beta), view_P,
            view_P, ElementAccumulator(0));

        // Compute softmax for P. We need to explicitly compute softmax
        // over P because softmax is fused to the second GEMM in the
        // profiled implementation.
        std::vector<ElementP> matrix_Ref(layout_P.capacity(extent_P));
        mutlass::device_memory::copy_to_host(
            matrix_Ref.data(), block_Ref_P.get(), matrix_Ref.size());
        mutlass::TensorView<ElementP, LayoutP> view_Ref_host(
            matrix_Ref.data(), layout_P, extent_P);

        std::vector<ElementS> matrix_Ref_S(layout_P.capacity(extent_P));
        for (int i = 0; i < matrix_Ref.size(); ++i) {
          matrix_Ref_S.at(i) = ElementS(matrix_Ref.at(i));
        }
        // Copy to S (for debugging).
        mutlass::device_memory::copy_to_device(offsetS, matrix_Ref_S.data(),
                                               matrix_Ref_S.size());
#else

        std::vector<ElementP> block_Ref_P(
            layout_P.capacity(extent_P));
        mutlass::TensorView<ElementP, LayoutP> view_P(block_Ref_P.data(),
                                                      layout_P, extent_P);
        mutlass::reference::host::GemmComplex<
            ElementQ, LayoutQ, ElementK, LayoutK, ElementP, LayoutP,
            ElementAccumulator, ElementCompute>(
            problem0, ElementAccumulator(alphaMma0), view_Q,
            mutlass::ComplexTransform::kNone, view_K,
            mutlass::ComplexTransform::kNone, ElementAccumulator(beta), view_P,
            view_P, ElementAccumulator(0));

        // Compute softmax for P. We need to explicitly compute softmax
        // over P because softmax is fused to the second GEMM in the
        // profiled implementation.
        std::vector<ElementP> matrix_Ref(layout_P.capacity(extent_P));
        mutlass::device_memory::copy_host_to_host(
            matrix_Ref.data(), block_Ref_P.data(), matrix_Ref.size());
        mutlass::TensorView<ElementP, LayoutP> view_Ref_host(
            matrix_Ref.data(), layout_P, extent_P);

        std::vector<ElementS> matrix_Ref_S(layout_P.capacity(extent_P));
        for (int i = 0; i < matrix_Ref.size(); ++i) {
          matrix_Ref_S.at(i) = ElementS(matrix_Ref.at(i));
        }
        // Copy to S (for debugging).
        mutlass::device_memory::copy_host_to_host(offsetS, matrix_Ref_S.data(),
                                                  matrix_Ref_S.size());

#endif
        if (causal) {
          for (int m = 0; m < problem0.m(); m++) {
            for (int n = 1; n < problem0.n(); n++) {
              if ((m / n) < (problem0.m() / problem0.n())) {
                view_Ref_host.ref().at({m, n}) += -std::numeric_limits<float>::infinity();
              }
            }
          }
        }
        
        if (with_softmax) {
          int n_dim_row = problem0.n();

          // Compute softmax for reference matrix
          if (usePow2) {
            for (int m = 0; m < problem0.m(); m++) {
              for (int n = 0; n < n_dim_row; n++) {
                view_Ref_host.ref().at({m, n}) =
                    kLog2e * postScaling * view_Ref_host.ref().at({m, n});
              }
              ElementSoftmaxCompute max =
                  ElementSoftmaxCompute(view_Ref_host.ref().at({m, uint64_t(0)}));
              for (int n = 1; n < n_dim_row; n++) {
                max = std::max(
                    max, ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})));
              }

              view_Norm_Ref.at({m, uint64_t(0)}) = ElementNorm(max);

              ElementSoftmaxCompute sum = ElementSoftmaxCompute();
              for (int n = 0; n < n_dim_row; n++) {
                sum += std::exp2f(
                    ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})) - max);
              }
              ElementSoftmaxCompute inv_sum = ElementSoftmaxCompute(1.0f / sum);

              view_Sum_Ref.at({m, uint64_t(0)}) = ElementSum(inv_sum);

              for (int n = 0; n < n_dim_row; n++) {
                view_Ref_host.ref().at({m, n}) =
                    ElementP(std::exp2f(ElementSoftmaxCompute(
                                            view_Ref_host.ref().at({m, n})) -
                                        max) *
                            inv_sum);
              }
            }
          } else {
            for (int64_t m = 0; m < problem0.m(); m++) {
              ElementSoftmaxCompute max =
                  ElementSoftmaxCompute(view_Ref_host.ref().at({m, uint64_t(0)}));
              for (int64_t n = 1; n < n_dim_row; n++) {
                max = std::max(
                    max, ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})));
              }

              view_Norm_Ref.at({m, uint64_t(0)}) = ElementNorm(max);

              ElementSoftmaxCompute sum = ElementSoftmaxCompute();
              for (int64_t n = 0; n < n_dim_row; n++) {
                sum += std::exp(
                    ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})) - max);
              }
              ElementSoftmaxCompute inv_sum = ElementSoftmaxCompute(1.0f / sum);

              view_Sum_Ref.at({m, uint64_t(0)}) = ElementSum(inv_sum);

              for (int64_t n = 0; n < n_dim_row; n++) {
                view_Ref_host.ref().at({m, n}) =
                    ElementP(std::exp(ElementSoftmaxCompute(
                                          view_Ref_host.ref().at({m, n})) -
                                      max) *
                            inv_sum);
              }
            }
          }
        }

#if DEVICE_REF
        mutlass::device_memory::copy_to_device(
            block_Ref_P.get(), matrix_Ref.data(), matrix_Ref.size());

        mutlass::reference::device::GemmComplex<
            ElementP, LayoutP, ElementV, LayoutV, ElementO, LayoutO,
            ElementAccumulator, ElementCompute>(
            problem1, ElementAccumulator(alpha1), view_P,
            mutlass::ComplexTransform::kNone, view_V,
            mutlass::ComplexTransform::kNone, ElementAccumulator(beta), view_O,
            view_O, ElementAccumulator(0));
#else

        mutlass::device_memory::copy_host_to_host(
            block_Ref_P.data(), matrix_Ref.data(), matrix_Ref.size());
        mutlass::reference::host::GemmComplex<
            ElementP, LayoutP, ElementV, LayoutV, ElementO, LayoutO,
            ElementAccumulator, ElementCompute>(
            problem1, ElementAccumulator(alpha1), view_P,
            mutlass::ComplexTransform::kNone, view_V,
            mutlass::ComplexTransform::kNone, ElementAccumulator(beta), view_O,
            view_O, ElementAccumulator(0));
#endif
      }
    }
  }
};
