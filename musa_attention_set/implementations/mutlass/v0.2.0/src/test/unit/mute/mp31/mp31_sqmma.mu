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

#include "mutlass_unit_test.h"

#include "mutlass/mutlass.h"
#include "mutlass/numeric_conversion.h"
#include "mutlass/gemm/gemm.h"
#include "mutlass/util/device_memory.h"
#include "mutlass/util/reference/host/tensor_fill.h"

#include <iostream>
#include <cstdlib>

#include <mute/tensor.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mute/atom/copy_atom.hpp>


using namespace mute;

namespace {

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

} // namespace


template <
  class TiledMma,
  class Traits,
  class ElementA,
  class ElementB,
  class ElementC,
  class SmemLayoutA,
  class SmemLayoutB>
__global__ __launch_bounds__(128) void sqmma_inst(ElementC* C, ElementA* A, ElementB* B) {
  TiledMma tiled_mma;

  static constexpr auto M = size<0>(typename TiledMma::AtomShape_MNK{});
  static constexpr auto N = size<1>(typename TiledMma::AtomShape_MNK{});
  static constexpr auto K = size<2>(typename TiledMma::AtomShape_MNK{});

  auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), typename Traits::AStride{});
  auto gB = make_tensor(make_gmem_ptr(B), make_shape(N, K), typename Traits::BStride{});
  auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), GenRowMajor{});

  auto accum = partition_fragment_C(tiled_mma, make_shape(M, N));
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);

  __shared__ ElementA __attribute__((aligned(256))) smem_a[M * K];
  __shared__ ElementB __attribute__((aligned(256))) smem_b[N * K];

  Tensor sA = make_tensor(make_smem_ptr(&smem_a[0]), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(&smem_b[0]), SmemLayoutB{});

  // LDGSTS
  auto tiled_copy_a = make_tiled_copy(Copy_Atom<DefaultCopy, ElementA>{},
                                      Layout<Shape<_16, _8>>{},
                                      Layout<Shape<_1, _1>>{});

  auto tiled_copy_b = make_tiled_copy(Copy_Atom<DefaultCopy, ElementB>{},
                                      Layout<Shape<_16, _8>>{},
                                      Layout<Shape<_1, _1>>{});

  auto thr_tile_copy_a = tiled_copy_a.get_thread_slice(threadIdx.x);
  auto thr_tile_copy_b = tiled_copy_b.get_thread_slice(threadIdx.x);

  Tensor tAgA = thr_tile_copy_a.partition_S(gA);
  Tensor tAsA = thr_tile_copy_a.partition_D(sA);

  Tensor tBgB = thr_tile_copy_b.partition_S(gB);
  Tensor tBsB = thr_tile_copy_b.partition_D(sB);

  copy(tiled_copy_a, tAgA, tAsA);
  copy(tiled_copy_b, tBgB, tBsB);

  __syncthreads();

  // SQMMA
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  mute::gemm(tiled_mma, tCrA, tCrB, accum);

  // STG
  copy(accum, tCgC);
}

///////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
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

  mutlass::reference::host::BlockFillRandomUniform(
    block, capacity, seed, scope_max, scope_min, 0);

  return true;
}

///////////////////////////////////////////////////////////////////////////////

template <class MMAOp, class Traits>
bool SqmmaInstRunner() {
  using TiledMma = TiledMMA<MMA_Atom<MMAOp>, Layout<Shape<_1,_1,_1>>>;

  using ElementA = typename TiledMma::ValTypeA;
  using ElementB = typename TiledMma::ValTypeB;
  using ElementC = typename TiledMma::ValTypeC;


  static constexpr auto M = size<0>(typename TiledMma::AtomShape_MNK{});
  static constexpr auto N = size<1>(typename TiledMma::AtomShape_MNK{});
  static constexpr auto K = size<2>(typename TiledMma::AtomShape_MNK{});

  using SmemLayoutA = decltype(
      MP31::SQMMA::make_canonical_gemm_smem_atom_layout<ElementA, Traits::AMajor, Int<M>, Int<K>>());
  using SmemLayoutB = decltype(
      MP31::SQMMA::make_canonical_gemm_smem_atom_layout<ElementB, Traits::BMajor, Int<N>, Int<K>>());

  mutlass::DeviceAllocation<ElementA> block_A;
  mutlass::DeviceAllocation<ElementB> block_B;
  mutlass::DeviceAllocation<ElementC> block_C;
  mutlass::DeviceAllocation<ElementC> block_C_ref;

  std::vector<ElementA> vector_A;
  std::vector<ElementB> vector_B;
  std::vector<ElementC> vector_C;
  std::vector<ElementC> vector_C_ref;

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


  sqmma_inst<TiledMma, Traits, ElementA, ElementB, ElementC, SmemLayoutA, SmemLayoutB>
              <<<1, 128>>>(block_C.get(), block_A.get(), block_B.get());

  musaError_t result = musaDeviceSynchronize();
  if (result != musaSuccess) {
    std::cerr << "Error running SQMMA INST UT. Last MUSA error is: "
              << musaGetErrorString(result) << std::endl;
    return false;
  }

  auto tA = make_tensor(vector_A.data(), make_shape(M, K), typename Traits::AStride{});
  auto tB = make_tensor(vector_B.data(), make_shape(N, K), typename Traits::BStride{});
  auto tC = make_tensor(vector_C_ref.data(), make_shape(M, N), GenRowMajor{});

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      ElementC acc = 0;
      for (int p = 0; p < K; ++p) {
        acc += ElementC(tA(i, p)) * ElementC(tB(j,p));
      }
      tC(i, j) = acc;
    }
  }

  block_C.copy_to_host(vector_C.data());

  musaDeviceSynchronize();

  for (int i = 0; i < M * N; ++i) {
    if (vector_C_ref[i] != vector_C[i]) {
      return false;
    }
  }
  return true;
}

TEST(MP31_MuTe_SQMMA, F32F16F16_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x32_F32F16F16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x32_F32F16F16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x32_F32F16F16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x32_F32F16F16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, F32BF16BF16_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x32_F32BF16BF16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x32_F32BF16BF16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x32_F32BF16BF16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x32_F32BF16BF16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, F32TF32TF32_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_64x64x8_F32TF32TF32_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_64x64x8_F32TF32TF32_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_64x64x8_F32TF32TF32_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_64x64x8_F32TF32TF32_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x8_F32TF32TF32_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, F32E5M2E5M2_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_F32E5M2E5M2_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_F32E5M2E5M2_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_F32E5M2E5M2_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_F32E5M2E5M2_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, F32E4M3E4M3_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_F32E4M3E4M3_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_F32E4M3E4M3_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_F32E4M3E4M3_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_F32E4M3E4M3_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, S32S8S8_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_S32S8S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_S32S8S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_S32S8S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_S32S8S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, U32U8U8_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_U32U8U8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_U32U8U8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_U32U8U8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x128x128_U32U8U8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, F32F16S8_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x32x64_F32F16S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x32x64_F32F16S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x32x64_F32F16S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x32x64_F32F16S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, F32BF16S8_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x32x64_F32BF16S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x32x64_F32BF16S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x32x64_F32BF16S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_128x32x64_F32BF16S8_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, F32S8F16_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_32x32x64_F32S8F16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_32x32x64_F32S8F16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_32x32x64_F32S8F16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_32x32x64_F32S8F16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}


TEST(MP31_MuTe_SQMMA, F32S8BF16_sqmma_inst_test) {

  // TT
  {
    using Traits = TT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_32x32x64_F32S8BF16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // TN
  {
    using Traits = TN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_32x32x64_F32S8BF16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NT
  {
    using Traits = NT_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_32x32x64_F32S8BF16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

  // NN
  {
    using Traits = NN_Traits;

    EXPECT_TRUE((SqmmaInstRunner<MP31_32x32x64_F32S8BF16_SS<Traits::AMajor, Traits::BMajor>, Traits>()));
  }

}

