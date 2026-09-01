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
#include "mutlass/util/device_memory.h"

#include <vector>

#include <mute/tensor.hpp>
#include <mute/atom/copy_atom.hpp>

using namespace mute;

namespace {

enum class TestKind {
  Load,
  Store,
  LoadWithPred,
  LdgstsWithPred,
  LoadWithDeviceDesc
};

template <class T,
          class TensorShape,
          class CopyPolicy,
          TestKind TestPolicy,
          class Desc>
__global__ void robust_buffer_access_test(const T* a_ptr, T* b_ptr, const Desc desc) {
  __shared__ T smem[4];
  auto gA = make_tensor(make_gmem_ptr(a_ptr), TensorShape{});
  auto sA = make_tensor(make_smem_ptr(smem), TensorShape{});
  auto gB = make_tensor(make_gmem_ptr(b_ptr), TensorShape{});

  CopyPolicy tile_copy;
  auto thr_copy = tile_copy.get_thread_slice(threadIdx.x);

  if constexpr (TestPolicy == TestKind::Load ||
                TestPolicy == TestKind::LoadWithPred) {
    auto tAgA = thr_copy.partition_S(gA);
    auto frag = make_fragment_like(tAgA);

    if constexpr (TestPolicy == TestKind::Load) {
      copy(tile_copy.with(desc), tAgA, frag);
    }
    else {
      auto pred = make_tensor<bool>(make_shape(size<1>(frag), size<2>(frag)), Stride<_1,_0>{});

      MUTE_UNROLL
      for (int i = 0; i < size(pred); ++i) {
        pred(i) = threadIdx.x == 1;
      }
      copy_if(tile_copy.with(desc), pred, tAgA, frag);
    }

    // musa stg
    MUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) {
      b_ptr[threadIdx.x * size(frag) + i] = frag[i];
    }
  }
  else if constexpr (TestPolicy == TestKind::Store) {
    auto tBgB = thr_copy.partition_D(gB);
    auto frag = make_fragment_like(tBgB);

    MUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) {
      frag(i) = 1;
    }

    copy(tile_copy.with(desc), frag, tBgB);
  }
  else if constexpr (TestPolicy == TestKind::LdgstsWithPred) {
    auto tAgA = thr_copy.partition_S(gA);
    auto tAsA = thr_copy.partition_D(sA);

    auto pred = make_tensor<bool>(make_shape(size<1>(tAgA), size<2>(tAgA)), Stride<_1,_0>{});

    MUTE_UNROLL
    for (int i = 0; i < size(pred); ++i) {
      pred(i) = threadIdx.x == 1;
    }

    copy_if(tile_copy.with(desc), pred, tAgA, tAsA);

    ldgsts_wait();
    __syncthreads();

    if (threadIdx.x == 0) {
      MUTE_UNROLL
      for (int i = 0; i < 4; ++i) {
        b_ptr[i] = smem[i];
      }
    }
  }
  else if constexpr (TestPolicy == TestKind::LoadWithDeviceDesc) {
    auto tAgA = thr_copy.partition_S(gA);
    auto frag = make_fragment_like(tAgA);

    auto device_desc = make_robust_desc(a_ptr, cosize(gA.layout()));
    copy(tile_copy.with(device_desc), tAgA, frag);
    // musa stg
    MUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) {
      b_ptr[threadIdx.x * size(frag) + i] = frag[i];
    }
  }
}

} // namespace

using Element = int;
constexpr int Alignment = 2;
using AlignmentType = mute::uint_bit_t<sizeof_bits_v<Element> * Alignment>;

using ThrLayout = Layout<Shape<_1, _2>>;
using ValLayout = Layout<Shape<_1, Int<Alignment>>>;

constexpr int ValidNum = 3;
using TensorShape = Shape<_1, Int<ValidNum>>;

TEST(MP31_MuTe_ROBUST_LSU, Load) {
  // 2 threads and each thread reads 2 integers
  // |<------   ValidNum  ------>|<-- OOB -->|
  // |---------------------------------------|
  // |  T0V0  |  T0V1  |  T1V0   |   T1V1    |
  // |---------------------------------------|

  using CopyPolicy = decltype(make_tiled_copy(
                  Copy_Atom<MP31_ROBUST_LOAD<AlignmentType>, Element>{},
                  ThrLayout{}, ValLayout{}));

  std::vector<Element> h_A(ValidNum);

  // valid int array: 1, 2, 3
  for (int i = 0; i < ValidNum; ++i) {
    h_A[i] = i + 1;
  }

  mutlass::DeviceAllocation<Element> d_A(ValidNum);
  d_A.copy_from_host(h_A.data());

  mutlass::DeviceAllocation<Element> d_B(4);

  auto robust_desc = make_robust_desc(d_A.get(), ValidNum);
  constexpr TestKind Kind = TestKind::Load;

  robust_buffer_access_test<Element, TensorShape, CopyPolicy, Kind, decltype(robust_desc)>
                            <<<1, 2>>>(d_A.get(), d_B.get(), robust_desc);

  std::vector<Element> h_B(4);

  d_B.copy_to_host(h_B.data());

  EXPECT_TRUE(h_B[0] == 1);
  EXPECT_TRUE(h_B[1] == 2);
  EXPECT_TRUE(h_B[2] == 3);
  EXPECT_TRUE(h_B[3] == 0); // OOB
}

TEST(MP31_MuTe_ROBUST_LSU, Store) {
  // 2 threads and each thread writes 2 integers
  // |<------   ValidNum  ------>|<-- OOB -->|
  // |---------------------------------------|
  // |  T0V0  |  T0V1  |  T1V0   |   T1V1    |
  // |---------------------------------------|

  using CopyPolicy = decltype(make_tiled_copy(
                  Copy_Atom<MP31_ROBUST_STORE<AlignmentType>, Element>{},
                  ThrLayout{}, ValLayout{}));

  mutlass::DeviceAllocation<Element> d_B(4);
  std::vector<Element> values(4, 2025);
  d_B.copy_from_host(values.data());
  auto robust_desc = make_robust_desc(d_B.get(), ValidNum);
  constexpr TestKind Kind = TestKind::Store;

  robust_buffer_access_test<Element, TensorShape, CopyPolicy, Kind, decltype(robust_desc)>
                            <<<1, 2>>>(nullptr, d_B.get(), robust_desc);

  std::vector<Element> h_B(4);

  d_B.copy_to_host(h_B.data());


  EXPECT_TRUE(h_B[0] == 1);
  EXPECT_TRUE(h_B[1] == 1);
  EXPECT_TRUE(h_B[2] == 1);
  EXPECT_TRUE(h_B[3] == 2025); // OOB
}

TEST(MP31_MuTe_ROBUST_LSU, LoadWithPred) {
  // 2 threads and each thread reads 2 integers,
  // but we only want thread 1 to do the load op
  // |<-- PredFalse -->|<- RD  ->|<-- OOB -->|
  // |---------------------------------------|
  // |  T0V0  |  T0V1  |  T1V0   |   T1V1    |
  // |---------------------------------------|

  using CopyPolicy = decltype(make_tiled_copy(
                  Copy_Atom<MP31_ROBUST_LOAD<AlignmentType>, Element>{},
                  ThrLayout{}, ValLayout{}));

  std::vector<Element> h_A(ValidNum);

  // valid int array: 1, 2, 3
  for (int i = 0; i < ValidNum; ++i) {
    h_A[i] = i + 1;
  }

  mutlass::DeviceAllocation<Element> d_A(ValidNum);
  d_A.copy_from_host(h_A.data());

  mutlass::DeviceAllocation<Element> d_B(4);

  auto robust_desc = make_robust_desc(d_A.get(), ValidNum);
  constexpr TestKind Kind = TestKind::LoadWithPred;

  robust_buffer_access_test<Element, TensorShape, CopyPolicy, Kind, decltype(robust_desc)>
                            <<<1, 2>>>(d_A.get(), d_B.get(), robust_desc);

  std::vector<Element> h_B(4);

  d_B.copy_to_host(h_B.data());

  EXPECT_TRUE(h_B[0] == 0); // pred false for t0
  EXPECT_TRUE(h_B[1] == 0); // pred false for t0
  EXPECT_TRUE(h_B[2] == 3);
  EXPECT_TRUE(h_B[3] == 0); // OOB

}

TEST(MP31_MuTe_ROBUST_LSU, LdgstsWithPred) {
  // 2 threads and each thread reads 2 integers into smem,
  // but we only want thread 1 to do the load op
  // |<-- PredFalse -->|<- RD  ->|<-- OOB -->|
  // |---------------------------------------|
  // |  T0V0  |  T0V1  |  T1V0   |   T1V1    |
  // |---------------------------------------|

  using CopyPolicy = decltype(make_tiled_copy(
                  Copy_Atom<MP31_ROBUST_LDGSTS<AlignmentType>, Element>{},
                  ThrLayout{}, ValLayout{}));

  std::vector<Element> h_A(ValidNum);

  // valid int array: 1, 2, 3
  for (int i = 0; i < ValidNum; ++i) {
    h_A[i] = i + 1;
  }

  mutlass::DeviceAllocation<Element> d_A(ValidNum);
  d_A.copy_from_host(h_A.data());

  mutlass::DeviceAllocation<Element> d_B(4);

  auto robust_desc = make_robust_desc(d_A.get(), ValidNum);
  constexpr TestKind Kind = TestKind::LdgstsWithPred;

  robust_buffer_access_test<Element, TensorShape, CopyPolicy, Kind, decltype(robust_desc)>
                            <<<1, 2>>>(d_A.get(), d_B.get(), robust_desc);

  std::vector<Element> h_B(4);

  d_B.copy_to_host(h_B.data());

  EXPECT_TRUE(h_B[0] == 0); // pred false for t0
  EXPECT_TRUE(h_B[1] == 0); // pred false for t0
  EXPECT_TRUE(h_B[2] == 3);
  EXPECT_TRUE(h_B[3] == 0); // OOB
}

TEST(MP31_MuTe_ROBUST_LSU, LoadWithDeviceDesc) {
  // 2 threads and each thread reads 2 integers(w/ device robust desc)
  // |<------   ValidNum  ------>|<-- OOB -->|
  // |---------------------------------------|
  // |  T0V0  |  T0V1  |  T1V0   |   T1V1    |
  // |---------------------------------------|
  using CopyPolicy = decltype(make_tiled_copy(
                  Copy_Atom<MP31_ROBUST_LOAD<AlignmentType>, Element>{},
                  ThrLayout{}, ValLayout{}));

  std::vector<Element> h_A(ValidNum);

  // valid int array: 1, 2, 3
  for (int i = 0; i < ValidNum; ++i) {
    h_A[i] = i + 1;
  }

  mutlass::DeviceAllocation<Element> d_A(ValidNum);
  d_A.copy_from_host(h_A.data());

  mutlass::DeviceAllocation<Element> d_B(4);

  auto robust_desc = make_robust_desc(d_A.get(), ValidNum);
  constexpr TestKind Kind = TestKind::LoadWithDeviceDesc;

  robust_buffer_access_test<Element, TensorShape, CopyPolicy, Kind, decltype(robust_desc)>
                            <<<1, 2>>>(d_A.get(), d_B.get(), robust_desc);

  std::vector<Element> h_B(4);

  d_B.copy_to_host(h_B.data());

  EXPECT_TRUE(h_B[0] == 1);
  EXPECT_TRUE(h_B[1] == 2);
  EXPECT_TRUE(h_B[2] == 3);
  EXPECT_TRUE(h_B[3] == 0); // OOB
}

