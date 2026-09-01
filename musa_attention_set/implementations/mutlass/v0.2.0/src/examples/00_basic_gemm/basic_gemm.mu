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

/*! \file
    \brief Basic GEMM example using the low-level MUTLASS API.

    This example demonstrates how to construct and invoke a GEMM Kernel based on the MUTLASS API.

    The MUTLASS Gemm template is instantiated in the function MutlassSgemmTT. This kernel computes
    the general matrix product(GEMM) using single-precision floating-point arithmetic and assumes
    all matrics have row-major layout.

    Aside from defining and launching the SGEMM kernel, this example also shows several MUTLASS
    utilities. These utilities are intended to be useful supporting components for managing tensor
    and memory allocations, initializing and comparing results, and computing reference output.

    MUTLASS utilities are defined in the directory `tools/util`, and definitions appear namespace
    `mutlass::` or an inner namespace therein. Operations in `mutlass::reference::` have both
    host-side and device-side implementations.

    Note that this example only demonstrates how to step-by-step define the template parameters
    required for Gemm Kernel using the low-level API, and does not aim for performance.
*/

#include "mute/tensor.hpp"
#include "mute/atom/mma_atom.hpp"
#include "mute/atom/copy_atom.hpp"

#include "mutlass/mutlass.h"
#include "mutlass/numeric_conversion.h"

#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/gemm/collective/collective_mma.hpp"
#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "mutlass/epilogue/collective/collective_epilogue.hpp"
#include "mutlass/epilogue/thread/linear_combination.h"

#include "mutlass/util/command_line.h"
#include "mutlass/util/device_memory.h"
#include "mutlass/util/packed_stride.hpp"
#include "mutlass/util/host_tensor.h"
#include "mutlass/util/reference/device/tensor_fill.h"
#include "mutlass/util/reference/device/tensor_compare.h"
#include "mutlass/util/reference/device/gett.hpp"

#include "helper.h"

using namespace mute;

namespace example
{

struct Options {
  bool help;

  mute::Shape<int, int, int, int> problem_size;
  float alpha;
  float beta;

  Options():
    help(false),
    problem_size({2048, 2048, 2048, 1}),
    alpha(1.0),
    beta(0.0) { }

  bool valid() const {
    return get<0>(problem_size) > 0
        && get<1>(problem_size) > 0
        && get<2>(problem_size) > 0
        && get<3>(problem_size) > 0;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    mutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", get<0>(problem_size));
    cmd.get_cmd_line_argument("n", get<1>(problem_size));
    cmd.get_cmd_line_argument("k", get<2>(problem_size));
    cmd.get_cmd_line_argument("batch_size", get<3>(problem_size));

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

  }

  // Prints the usage statement
  std::ostream & print_usage(std::ostream &out) const {

    out <<
      "00_basic_gemm example\n"
      "\n"
      " This example uses MUTLASS to run a SGEMM Kernel\n"
      "\n"
      "Options:\n"
      "  --help                      If specified, displays this usage statement.\n"
      "  --m=<int>                   GEMM M dimension\n"
      "  --n=<int>                   GEMM N dimension\n"
      "  --k=<int>                   GEMM K dimension\n"
      "  --batch_size=<int>          GEMM batch dimension\n"
      "  --alpha=<float>             GEMM alpha parameter\n"
      "  --beta=<float>              GEMM beta parameter\n"
      "\n"
      "Examples:\n"
      "\n"
      "$ ./examples/00_basic_gemm/00_basic_gemm --m=4096 --n=2048 --k=3072 --batch_size=3\n";

    return out;
  }
};


} // namespace example

///////////////////////////////////////////////////////////////////////////////////////////////////

// Define the layout of matrix A, B and C
using LayoutA = mutlass::layout::RowMajor;
using LayoutB = mutlass::layout::RowMajor;
using LayoutC = mutlass::layout::RowMajor;

// Define the data type of matrix A, B and C
using TypeA = float;
using TypeB = float;
using TypeC = float;

// Define the TiledMma:
// 1. MMA_Atom uses FMA and assumes the types involved are the same as the matrix types
// 2. Expand MMA_Atom to 16x16x1 in the M, N and K respectively
using TiledMma = TiledMMA<MMA_Atom<UniversalFMA<TypeC, TypeA, TypeB>>,
                          Layout<Shape<_16, _16, _1>>>;

// Define TileShape
using TileShape = Shape<_128, _128, _4>;

static constexpr int AlignmentA = 1;
static constexpr int AlignmentB = 1;

// Define Dispatch Policy, and it is recommended to use MainloopMp22TwoStageUnpredicated
// if there is no need to deal with boundary
using DispatchPolicy = mutlass::gemm::MainloopMp22TwoStage;

// Define Gmem TiledCopy:
// Maximize the number of threads along the gmem major mode to promote coalesced reads
// For Row-major Matrix A(MxK), the major mode is K
using GmemCopyOpA     = UniversalCopy<mute::uint_bit_t<AlignmentA * sizeof_bits_v<TypeA>>>;
using GmemTiledCopyA  = decltype(make_tiled_copy(
                          Copy_Atom<GmemCopyOpA, TypeA>{},
                          Layout<Shape<_64, _4>, Stride< _4, _1>>{}, // threads layout 64x4
                          Layout<Shape<_1, Int<AlignmentA>>>{}));    // value   layout 1xAlignmentA
// For Row-major Matrix B(NxK), the major mode is N
using GmemCopyOpB     = UniversalCopy<mute::uint_bit_t<AlignmentB * sizeof_bits_v<TypeB>>>;
using GmemTiledCopyB  = decltype(make_tiled_copy(
                          Copy_Atom<GmemCopyOpB, TypeB>{},
                          Layout<Shape<_128, _2>, Stride<_1, _128>>{}, // threads layout 128x2
                          Layout<Shape<Int<AlignmentB>, _1>>{}));      // value   layout AlignmentBx1

// Define Smem Layout
using SmemLayoutAtomA = Layout<Shape<_128, _4>, Stride<_1, _128>>;
using SmemCopyAtomA   = Copy_Atom<DefaultCopy, TypeA>;

using SmemLayoutAtomB = Layout<Shape<_128, _4>>;
using SmemCopyAtomB   = Copy_Atom<DefaultCopy, TypeB>;


// Define CollectiveMainloop
using CollectiveMainloop = mutlass::gemm::collective::CollectiveMma<
  DispatchPolicy, TileShape,
  TypeA, mutlass::detail::TagToStrideA_t<LayoutA>,
  TypeB, mutlass::detail::TagToStrideB_t<LayoutB>,
  TiledMma,
  GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, mute::identity,  // A
  GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, mute::identity   // B
>;

// Define CollectiveEpilogue
using CollectiveEpilogue = mutlass::epilogue::collective::DefaultEpilogue<
  mutlass::detail::TagToStrideC_t<LayoutC>,
  mutlass::detail::TagToStrideC_t<LayoutC>,
  mutlass::epilogue::thread::LinearCombination<TypeC, 1, TypeC, TypeC>,
  mutlass::gemm::EpilogueDefault>;


// Define Kernel-level gemm type
using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
  Shape<int, int, int, int>,  // problem_size
  CollectiveMainloop,
  CollectiveEpilogue
>;

// Define Device-level gemm type
using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Instantiate and launch a GEMM kernel
int MutlassSgemmTT(
  int M,
  int N,
  int K,
  int Batch,
  float alpha,
  float const *A,
  float const *B,
  float beta,
  float *C,
  float *C_ref) {
  // Get ProblemSize type
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  ProblemShapeType problem_size {M, N, K, Batch};

  // Get Stride type
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;

  // Assume the matrics are packed
  StrideA stride_a = mutlass::make_mute_packed_stride(StrideA{}, mute::make_shape(M, K, Batch));
  StrideB stride_b = mutlass::make_mute_packed_stride(StrideB{}, mute::make_shape(N, K, Batch));
  StrideC stride_c = mutlass::make_mute_packed_stride(StrideC{}, mute::make_shape(M, N, Batch));

  // Init gemm arguments
  auto arguments = typename Gemm::Arguments {
    mutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    {A, stride_a, B, stride_b},                 // mainloop arguments
    {{alpha, beta}, C, stride_c, C, stride_c},  // epilogue arguments
    mutlass::KernelHardwareInfo{}
  };

  // Instantiate MUTLASS kernel depending on templates
  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  mutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  MUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize MUTLASS kernel with arguments and workspace pointer
  MUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Run Gemm Kernel
  MUTLASS_CHECK(gemm.run());

  auto musa_err = musaDeviceSynchronize();
  if (musaSuccess != musa_err) {
    std::cerr << "ERROR: SGEMM operator execution failed. with error :";
    std::cerr << musaGetErrorString(musa_err) << "\n";
    return 1;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////

  //
  // Verify
  //

  mutlass::reference::device::gett(
    problem_size,
    A, stride_a,
    B, stride_b,
    float{},
    C_ref, stride_c,
    C_ref, stride_c,
    alpha, beta);

  musa_err = musaDeviceSynchronize();
  if (musaSuccess != musa_err) {
    std::cerr << "ERROR: SGEMM reference execution failed. with error :";
    std::cerr << musaGetErrorString(musa_err) << "\n";
    return 1;
  }

  return 0;
}


int TestMutlassGemm(Shape<int, int, int, int> problem, float alpha, float beta) {
  auto [M, N, K, Batch] = problem;

  std::vector<TypeA> h_A(M * K * Batch);
  std::vector<TypeB> h_B(N * K * Batch);
  std::vector<TypeC> h_C(M * N * Batch);

  for (auto& a : h_A) a = TypeA(int(4*(rand() / double(RAND_MAX)) - 1));
  for (auto& b : h_B) b = TypeB(int(4*(rand() / double(RAND_MAX)) - 1));
  for (auto& c : h_C) c = TypeC(int(4*(rand() / double(RAND_MAX)) - 1));

  // adhoc for 3D tensor
  mutlass::HostTensor<TypeA, LayoutA> A({M, K * Batch});
  mutlass::HostTensor<TypeB, LayoutB> B({N, K * Batch});
  mutlass::HostTensor<TypeC, LayoutC> C_mutlass({M, N * Batch});
  mutlass::HostTensor<TypeC, LayoutC> C_reference({M, N * Batch});


  A.copy_in_host_to_device(h_A.data());
  B.copy_in_host_to_device(h_B.data());
  C_mutlass.copy_in_host_to_device(h_C.data());
  C_reference.copy_in_host_to_device(h_C.data());

  auto status = MutlassSgemmTT(
      M,
      N,
      K,
      Batch,
      alpha,
      A.device_data(),
      B.device_data(),
      beta,
      C_mutlass.device_data(),
      C_reference.device_data());

  if (0 != status) {
    std::cerr << "ERROR when running test\n";
    return 1;
  }

  // Compare
  bool passed = mutlass::reference::device::BlockCompareEqual(
      C_mutlass.device_data(),
      C_reference.device_data(),
      M * N * Batch);

  if (passed) {
    std::cout << "MUTLASS SGEMM verification passed.\n";
    return 0;
  } else {
    std::cerr << "ERROR: MUTLASS SGEMM verification failed.\n";
    return 1;
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
int main(int argc, const char* arg[]) {
  example::Options options;
  options.parse(argc, arg);

  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return EXIT_SUCCESS;
  }

  return TestMutlassGemm(options.problem_size, options.alpha, options.beta);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
