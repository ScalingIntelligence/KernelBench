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
    \brief Quyuan GEMM example leveraging collective operation builders.

    This example showcases the use of MUTLASS's CollectiveBuilder to easily construct performant kernels
    targeting the MooreThreads Quyuan architecture.

    MUTLASS's CollectiveBuilder tries to ease the process of selecting template parameters for kernels.
    The CollectiveBuilder only takes in a small set of template parameters. It then automatically determines
    the template parameters such as the MMA Operation, Shared Memory Layout, and Global Memory Read strategy
    based on generic properties of the operands, layout, and other parameters. For example, when the permute
    layout is set to `Auto`, the CollectiveBuilder may automatically calculate the best PermuteLayout for
    TiledMma given the threads count and the thread block shape.

    MUTLASS builders make an attempt to pick the best template parameters so that the assembled collectives
    have the best performance, but this is not a guarantee. A user relying on `Auto` may get a free performance
    upgrade with newer MUTLASS releases in case we can provide more optimized implementations that the builder
    can transparently assemble for `Auto`. However a user should not rely on `Auto` if they require a specific
    permute layout and/or scheduling policy to be used.

    If a user decides to let the builders pick the collective specialization via `Auto`-type, they should be
    used for both mainloop and epilogue alike to ensure compatibility between the chosen collectives.

    Example usage:
      $ ./examples/01_quyuan_gemm_with_collective_builder/01_collective_builder \
            --m=2048 --n=2048 --k=1024 --l=2
*/

#include <iostream>

#include "mute/tensor.hpp"

#include "mutlass/mutlass.h"

#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/epilogue/collective/collective_builder.hpp"

#include "mutlass/util/command_line.h"
#include "mutlass/util/device_memory.h"
#include "mutlass/util/packed_stride.hpp"
#include "mutlass/util/host_tensor.h"
#include "mutlass/util/reference/device/tensor_fill.h"
#include "mutlass/util/reference/device/tensor_compare.h"
#include "mutlass/util/reference/device/gett.hpp"

#include "helper.h"

using namespace mute;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
namespace example
{

struct Options {

  bool help;

  int m, n, k, l;
  float alpha, beta;

  Options():
    help(false),
    m(2048), n(2048), k(2048), l(1),
    alpha(1.f), beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    mutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 2048);
    cmd.get_cmd_line_argument("n", n, 2048);
    cmd.get_cmd_line_argument("k", k, 2048);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "01_quyuan_gemm_with_collective_builder\n\n"
      << "  This example showcases the use of MUTLASS's collective operation builders to easily construct\n"
      << "  performant kernels targeting MooreThreads's Quyuan architecture.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n";

    return out;
  }
};

} // namespace example

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  mutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = mutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else {
    scope_max = 8;
    scope_min = -8;
  }

  mutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char* arg[]) {
  musaDeviceProp props;

  MUSA_CHECK(musaGetDeviceProperties(&props, 0));

  if (props.major * 10 + props.minor != 22) {
    std::cout
      << "This example requires a GPU of MooreThreads's Quyuan Architecture.\n";
    return 0;
  }

  example::Options options;
  options.parse(argc, arg);

  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return EXIT_SUCCESS;
  }

  //
  // Build Gemm Kernel
  //
  using LayoutA = mutlass::layout::ColumnMajor;
  using LayoutB = mutlass::layout::RowMajor;
  using LayoutC = mutlass::layout::ColumnMajor;
  using LayoutD = mutlass::layout::ColumnMajor;

  using ElementA = mutlass::half_t;
  using ElementB = mutlass::half_t;
  using ElementC = mutlass::half_t;
  using ElementD = mutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;

  using ArchTag = mutlass::arch::Mp22;
  using OpClass = mutlass::arch::OpClassTensorOp;

  // 16Byte Alignment
  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using CollectiveMainloop = typename mutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass,
      ElementA, LayoutA, AlignmentA,                        // Operand A
      ElementB, LayoutB, AlignmentB,                        // Operand B
      ElementAccumulator,
      Shape<_256, _128, _32>,                               // TileShape
      Shape<_1, _1, _1>,                                    // ClusterShape
      mutlass::gemm::collective::StageCountAuto,            // StageCountType
      mutlass::gemm::collective::KernelScheduleAuto         // KernelScheduleType
  >::CollectiveOp;

  using CollectiveEpilogue = typename mutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OpClass,
      Shape<_256, _128, _32>,                               // TileShape
      Shape<_1, _1, _1>,                                    // ClusterShape
      mutlass::epilogue::collective::EpilogueTileAuto,      // EpilogueTileType
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,                        // Operand C
      ElementD, LayoutD, AlignmentD,                        // Output  D
      mutlass::epilogue::collective::EpilogueScheduleAuto   // EpilogueScheduleType
  >::CollectiveOp;

  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  //
  // Initialize operands
  //
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

  auto [M, N, K, L] = problem_size;
  uint64_t seed = 0;

  StrideA stride_A = mutlass::make_mute_packed_stride(StrideA{}, mute::make_shape(M, K, L));
  StrideB stride_B = mutlass::make_mute_packed_stride(StrideB{}, mute::make_shape(N, K, L));
  StrideC stride_C = mutlass::make_mute_packed_stride(StrideC{}, mute::make_shape(M, N, L));
  StrideD stride_D = mutlass::make_mute_packed_stride(StrideD{}, mute::make_shape(M, N, L));

  mutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  mutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  mutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  mutlass::DeviceAllocation<typename Gemm::ElementD> block_D;
  mutlass::DeviceAllocation<typename Gemm::ElementD> block_ref_D;


  block_A.reset(M * K * L);
  block_B.reset(K * N * L);
  block_C.reset(M * N * L);
  block_D.reset(M * N * L);
  block_ref_D.reset(M * N * L);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);


  typename Gemm::Arguments arguments {
    mutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta},
     block_C.get(), stride_C, block_D.get(), stride_D},
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
    std::cerr << "ERROR: GEMM operator execution failed. with error :";
    std::cerr << musaGetErrorString(musa_err) << "\n";
    return 1;
  }

  //
  // Verify
  //
  mutlass::reference::device::gett(
    problem_size,
    block_A.get(), stride_A,
    block_B.get(), stride_B,
    ElementAccumulator{},
    block_C.get(), stride_C,
    block_ref_D.get(), stride_D,
    options.alpha, options.beta);

  musa_err = musaDeviceSynchronize();
  if (musaSuccess != musa_err) {
    std::cerr << "ERROR: GEMM reference execution failed. with error :";
    std::cerr << musaGetErrorString(musa_err) << "\n";
    return 1;
  }

  // Compare
  bool passed = mutlass::reference::device::BlockCompareEqual(
      block_D.get(),
      block_ref_D.get(),
      block_D.size());

  if (passed) {
    std::cout << "MUTLASS GEMM verification passed.\n";
    return 0;
  } else {
    std::cerr << "ERROR: MUTLASS GEMM verification failed.\n";
    return 1;
  }
}
