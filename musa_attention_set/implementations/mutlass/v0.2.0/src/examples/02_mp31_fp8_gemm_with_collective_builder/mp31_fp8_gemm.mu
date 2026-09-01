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
    \brief Simple MP31 GEMM example using MUTLASS APIs for MooreThreads MP31 architecture

    This example demonstrate a simple way to instantiate and run a FP8 GEMM using the MUTLASS
    APIs on MooreThreads MP31 architecture. New features that will be showcased in this example are as follows:

    1. MooreThreads MP31 architecture introduces a new series of tensor core instructions (SQMMA)
    which are more efficient than the Quyuan tensor core instructions.

    2. MooreThreads MP31 architecture includes new Tensor Memory Engine (TME) unit to transfer large
    blocks of data efficiently between global memory and shared memory.

    3. A simple way to tune the CTA rasterization direction and swizzle pattern of MP31 kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.

    Examples:

      $ ./examples/02_mp31_fp8_gemm_with_collective_builder/02_mp31_fp8_gemm --m=2048 --n=2048 --k=2048 --rasterization=N --swizzle=2
*/

#include <iostream>

#include "mutlass/mutlass.h"

#include "mute/tensor.hpp"
#include "mutlass/tensor_ref.h"
#include "mutlass/epilogue/collective/default_epilogue.hpp"
#include "mutlass/epilogue/thread/linear_combination.h"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/epilogue/collective/collective_builder.hpp"
#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "mutlass/gemm/kernel/gemm_universal.hpp"
#include "mutlass/gemm/kernel/tile_scheduler_params.hpp"

#include "mutlass/util/command_line.h"
#include "mutlass/util/distribution.h"
#include "mutlass/util/host_tensor.h"
#include "mutlass/util/packed_stride.hpp"
#include "mutlass/util/tensor_view_io.h"
#include "mutlass/util/reference/device/gemm.h"
#include "mutlass/util/reference/device/tensor_compare.h"
#include "mutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace mute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = mutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = mutlass::layout::ColumnMajor;                   // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / mutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = mutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = mutlass::layout::RowMajor;                      // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / mutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = mutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = mutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / mutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ElementCompute      = float;                                           // Element type for epilogue computation
using ArchTag             = mutlass::arch::Mp31;                             // Tag indicating the minimum MP that supports the intended feature
using OperatorClass       = mutlass::arch::OpClassTensorOp;                  // Operator class tag
using TileShape           = Shape<_384,_256,_64>;                            // Threadblock-level tile size
using ClusterShape        = Shape<_1,_1,_1>;                                 // Shape of the threadblocks in a cluster
using StageCountType      = mutlass::gemm::collective::StageCount<4>;        // Stage count
using KernelSchedule      = mutlass::gemm::KernelTme;                        // Kernel to launch
using EpilogueSchedule    = mutlass::epilogue::WithTme;                      // Epilogue to launch
using ThreadEpilogueOp    = mutlass::epilogue::fusion::LinearCombination<ElementC,ElementCompute,ElementC,ElementCompute>;

using CollectiveMainloop = typename mutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    StageCountType,
    KernelSchedule
  >::CollectiveOp;

using CollectiveEpilogue = typename mutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    mutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    EpilogueSchedule,
    ThreadEpilogueOp,
    CollectiveMainloop
  >::CollectiveOp;

using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference = mutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

mutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
mutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
mutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
mutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
mutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename mutlass::gemm::kernel::detail::TileSchedulerDefaultParams::RasterOrderOptions;

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  RasterOrderOptions raster;
  int swizzle;

  Options():
    help(false),
    m(16032), n(16384), k(8192),
    alpha(1.f), beta(0.f),
    iterations(1),
    raster(RasterOrderOptions::Heuristic),
    swizzle(2)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    mutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h') {
      raster = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 2);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "02_mp31_fp8_gemm\n\n"
      << "  MP31 FP8 GEMM.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
      << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "02_collective_builder" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  mutlass::Status status;
  musaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    mutlass::Status status = mutlass::Status::kSuccess,
    musaError_t error = musaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  mutlass::DeviceAllocation<Element>& block,
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

  mutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = mutlass::make_mute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = mutlass::make_mute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = mutlass::make_mute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = mutlass::make_mute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    mutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}
  };

  arguments.scheduler.raster_order = options.raster;
  arguments.scheduler.swizzle_size = options.swizzle;

  return arguments;
}

bool verify(const Options &options) {
  mutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  mutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  mutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  mutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha),
    ref_A,
    ref_B,
    ElementAccumulator(options.beta),
    ref_C,
    ref_D);

  // Wait for kernel to finish
  MUSA_CHECK(musaDeviceSynchronize());

  // Check if output from MUTLASS kernel and reference kernel are equal or not
  bool passed = mutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

  return passed;
}

/// Exemute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate MUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  mutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  MUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize MUTLASS kernel with arguments and workspace pointer
  MUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  MUTLASS_CHECK(gemm.run());

  auto musa_err = musaDeviceSynchronize();
  if (musaSuccess != musa_err) {
    std::cerr << "ERROR: GEMM operator execution failed. with error :";
    std::cerr << musaGetErrorString(musa_err) << "\n";
    return 1;
  }

  // Check if output from MUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      MUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      MUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  musaDeviceProp props;
  int current_device_id;
  MUSA_CHECK(musaGetDevice(&current_device_id));
  MUSA_CHECK(musaGetDeviceProperties(&props, current_device_id));
  musaError_t error = musaGetDeviceProperties(&props, 0);
  if (props.major < 3) {
    std::cerr
      << "This example requires a GPU of MooreThreads's MP31 Architecture or "
      << "later (compute capability 31 or greater).\n";
    return 0;
  }
  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate MUTLASS kernels
  //
  run<Gemm>(options);

  block_A.reset();
  block_B.reset();
  block_C.reset();
  block_D.reset();
  block_ref_D.reset();

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
