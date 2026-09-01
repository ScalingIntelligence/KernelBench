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
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <musa_runtime_api.h>
#include <mute/tensor.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mute/atom/copy_atom.hpp>
#include "mutlass/mutlass.h"
#include "mutlass/numeric_conversion.h"
#include "mutlass/gemm/gemm.h"
#include "mutlass/util/device_memory.h"
#include "mutlass/util/reference/host/tensor_fill.h"

using namespace mute;
struct TT_Traits {
  using AStride = GenRowMajor;
  using BStride = GenColMajor;
  static constexpr TCE::Major AMajor = TCE::Major::K;
  static constexpr TCE::Major BMajor = TCE::Major::MN;
};

struct TN_Traits {
  using AStride = GenRowMajor;
  using BStride = GenRowMajor;
  static constexpr TCE::Major AMajor = TCE::Major::K;
  static constexpr TCE::Major BMajor = TCE::Major::K;
};

struct NT_Traits {
  using AStride = GenColMajor;
  using BStride = GenColMajor;
  static constexpr TCE::Major AMajor = TCE::Major::MN;
  static constexpr TCE::Major BMajor = TCE::Major::MN;
};

struct NN_Traits {
  using AStride = GenColMajor;
  using BStride = GenRowMajor;
  static constexpr TCE::Major AMajor = TCE::Major::MN;
  static constexpr TCE::Major BMajor = TCE::Major::K;
};

template <class MMA_Op>
struct StrideTraits;

template <
  class TiledMma,
  class OpStride,
  class SrcTypeA,
  class SrcTypeB,
  class AccType>
__global__ void mma_inst(AccType* C, SrcTypeA* A, SrcTypeB* B) {
  TiledMma tiled_mma;
  using GmemCopyAtomA = Copy_Atom<DefaultCopy, SrcTypeA>;
  using GmemCopyAtomB = Copy_Atom<DefaultCopy, SrcTypeB>;

  auto M = size<0>(typename TiledMma::AtomShape_MNK{});
  auto N = size<1>(typename TiledMma::AtomShape_MNK{});
  auto K = size<2>(typename TiledMma::AtomShape_MNK{});

  auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), typename OpStride::AStride{});
  auto gB = make_tensor(make_gmem_ptr(B), make_shape(N, K), typename OpStride::BStride{});
  auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), GenRowMajor{});


  auto accum = partition_fragment_C(tiled_mma, make_shape(M, N));
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCrA  = thr_mma.partition_fragment_A(gA);
  Tensor tCrB  = thr_mma.partition_fragment_B(gB);
  Tensor tCgC  = thr_mma.partition_C(gC);

  auto thr_copy_A       = make_tiled_copy_A(GmemCopyAtomA{}, tiled_mma).get_thread_slice(threadIdx.x);
  Tensor tCgA           = thr_copy_A.partition_S(gA);
  Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);

  auto thr_copy_B       = make_tiled_copy_B(GmemCopyAtomB{}, tiled_mma).get_thread_slice(threadIdx.x);
  Tensor tCgB           = thr_copy_B.partition_S(gB);
  Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

  copy(tCgA, tCrA_copy_view);
  copy(tCgB, tCrB_copy_view);
  mute::gemm(tiled_mma, accum, tCrA, tCrB, accum);

  copy(accum, tCgC);
}

template <class Element>
bool initialize_block(
  Element* block,
  size_t capacity,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = mutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  mutlass::reference::host::BlockFillSequential(
      block, capacity);

  return true;
}

template <class MmaOp, class Traits>
bool mma_test_body() {
  using TiledMma = TiledMMA<MMA_Atom<MmaOp>, Layout<Shape<_1, _1, _1>>>;
  using Atom_MNK = typename TiledMma::AtomShape_MNK;
  using AType = typename TiledMma::FrgTypeA;
  using BType = typename TiledMma::FrgTypeB;
  using CType = typename TiledMma::FrgTypeC;
  using DType = typename TiledMma::FrgTypeD;
  constexpr int32_t warp_size = size(TiledMma{});
  if constexpr (warp_size == 128) {
    static_assert(std::is_same<AType, BType>::value);
  }
  static_assert(std::is_same<CType, DType>::value);

  auto M = size<0>(Atom_MNK{});
  auto N = size<1>(Atom_MNK{});
  auto K = size<2>(Atom_MNK{});

  std::vector<AType> vector_A;
  std::vector<BType> vector_B;
  std::vector<CType> vector_C;
  std::vector<CType> vector_C_ref;

  mutlass::DeviceAllocation<AType> block_A;
  mutlass::DeviceAllocation<BType> block_B;
  mutlass::DeviceAllocation<CType> block_C;
  mutlass::DeviceAllocation<DType> block_C_ref;

  block_A.reset(M * K);
  block_B.reset(N * K);
  block_C.reset(M * N);
  block_C_ref.reset(M * N);

  vector_A.resize(M * K);
  vector_B.resize(N * K);
  vector_C.resize(M * N);
  vector_C_ref.resize(M * N);

  uint64_t seed = 1234;
  initialize_block(vector_A.data(), vector_A.capacity(), seed + 2024);
  initialize_block(vector_B.data(), vector_B.capacity(), seed + 2023);

  block_A.copy_from_host(vector_A.data());
  block_B.copy_from_host(vector_B.data());
  block_C.copy_from_host(vector_C.data());
  block_C_ref.copy_from_host(vector_C_ref.data());

  auto tA = make_tensor(vector_A.data(), make_shape(M, K), typename Traits::AStride{});
  auto tB = make_tensor(vector_B.data(), make_shape(N, K), typename Traits::BStride{});
  auto tC = make_tensor(vector_C_ref.data(), make_shape(M, N), GenRowMajor{});
  auto tC_ = make_tensor(vector_C.data(), make_shape(M, N), GenRowMajor{});


  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      CType acc = 0;
      for (int p = 0; p < K; ++p) {
        acc += CType(tA(i, p)) * CType(tB(j,p));
      }
      tC(i, j) = acc;
    }
  }

  mma_inst<TiledMma, Traits><<<1, warp_size>>>(block_C.get(), block_A.get(), block_B.get());

  musaError_t result = musaDeviceSynchronize();
  if (result != musaSuccess) {
    std::cerr << "Error running RR MMA INST UT. Last MUSA error is: "
              << musaGetErrorString(result) << std::endl;
    return false;
  }

  block_C.copy_to_host(vector_C.data());

  musaDeviceSynchronize();

  for (int i = 0; i < M * N; ++i) {
    EXPECT_EQ(tC_(i), tC(i));
  }
  return true;
}