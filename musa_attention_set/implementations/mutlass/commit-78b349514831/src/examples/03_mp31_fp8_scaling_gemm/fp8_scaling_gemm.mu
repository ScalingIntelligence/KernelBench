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
#include "mutlass/util/reference/host/tensor_fill.h"
#include "mutlass/util/reference/host/tensor_copy.h"
#include "mutlass/util/reference/host/tensor_compare.h"
#include "mutlass/util/reference/host/tensor_norm.h"

#include "reference/host/gemm_with_groupwise_scaling.hpp"

#include "helper.h"

using namespace mute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = mutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = mutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / mutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = mutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = mutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / mutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = float;                                          // Element type for C and D matrix operands
using         LayoutC     = mutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / mutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;


// Core kernel configurations
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ElementBlockScale   = float;
using ElementCompute      = float;                                           // Element type for epilogue computation

using TileShapeMNK        = Shape<_128,_256,_128>;                           // Threadblock-level tile size

using StageCountType      = mutlass::gemm::collective::StageCount<2>;       // Stage count


// ScaleGranularityM/N: number of rows in A/columns in B that share the same scaling factor
// ScaleGranularityK: number of columns in A & rows in B that share the same scaling factor
template <
  int ScaleGranularityM_,
  int ScaleGranularityN_,
  int ScaleGranularityK_>
struct GroupScaleConfig {
  using ArchTag        = mutlass::arch::Mp31;                             // Tag indicating the minimum MP that supports the intended feature
  using OperatorClass  = mutlass::arch::OpClassTensorOp;                  // Operator class tag
  using TileShape      = TileShapeMNK;
  using ClusterShape   = Shape<_1,_1,_1>;                                 // Shape of the threadblocks in a cluster

  static constexpr int ScaleGranularityM = ScaleGranularityM_;
  static constexpr int ScaleGranularityN = ScaleGranularityN_;
  static constexpr int ScaleGranularityK = ScaleGranularityK_;

  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;
  static constexpr int ScalePromotionInterleave = ScaleGranularityK / size<2>(TileShape{});

  static_assert(size<0>(TileShape{}) == ScaleGranularityM * ScaleMsPerTile,
              "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert(size<1>(TileShape{}) == ScaleGranularityN * ScaleNsPerTile,
              "FP8 scaling granularity must evenly divide tile shape along N.");
  static_assert(size<2>(TileShape{}) * ScalePromotionInterleave == ScaleGranularityK,
              "FP8 scaling granularity must be a multiple of the tile shape alone K.");

  using KernelSchedule      = mutlass::gemm::KernelTmeWarpSpecializedScaledAccum<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;                        // Kernel to launch
  using EpilogueSchedule    = mutlass::epilogue::NoSmem;                      // Epilogue to launch
  using ThreadEpilogueOp    = mutlass::epilogue::fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>;
};


using GroupScale1x128x128Config   = GroupScaleConfig<  1, 128, 128>;
using GroupScale128x128x128Config = GroupScaleConfig<128, 128, 128>;
using GroupScale256x256x256Config = GroupScaleConfig<256, 256, 256>;


template <
  class ScheduleConfig
>
struct GroupScaleGemm {
  using ArchTag           = typename ScheduleConfig::ArchTag;
  using OperatorClass     = typename ScheduleConfig::OperatorClass;
  using TileShape         = typename ScheduleConfig::TileShape;
  using ClusterShape      = typename ScheduleConfig::ClusterShape;
  using KernelSchedule    = typename ScheduleConfig::KernelSchedule;
  using EpilogueSchedule  = typename ScheduleConfig::EpilogueSchedule;
  using ThreadEpilogueOp  = typename ScheduleConfig::ThreadEpilogueOp;


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
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    ThreadEpilogueOp,
    CollectiveMainloop
  >::CollectiveOp;


  using GemmKernel = mutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    mutlass::gemm::PersistentScheduler
  >;

  using Gemm = mutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using GroupScale1x128x128Gemm = GroupScaleGemm<GroupScale1x128x128Config>;

// Extract information from Gemm kernel.
using EpilogueOutputOp  = typename GroupScale1x128x128Gemm::Gemm::EpilogueOutputOp;
using ElementScalar     = typename EpilogueOutputOp::ElementScalar;

using StrideA = typename GroupScale1x128x128Gemm::Gemm::GemmKernel::StrideA;
using StrideB = typename GroupScale1x128x128Gemm::Gemm::GemmKernel::StrideB;
using StrideC = typename GroupScale1x128x128Gemm::Gemm::GemmKernel::StrideC;
using StrideD = typename GroupScale1x128x128Gemm::Gemm::GemmKernel::StrideD;

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

//
// Data members
//

mutlass::HostTensor<ElementA, LayoutA> tensor_A;
mutlass::HostTensor<ElementB, LayoutB> tensor_B;
mutlass::HostTensor<ElementC, LayoutC> tensor_C;
mutlass::HostTensor<ElementD, LayoutD> tensor_D;

mutlass::HostTensor<ElementBlockScale, LayoutA> blockscale_tensor_A;
mutlass::HostTensor<ElementBlockScale, LayoutB> blockscale_tensor_B;
mutlass::HostTensor<ElementD, LayoutD> tensor_ref_D;

using RasterOrderOptions = typename mutlass::gemm::kernel::detail::TileSchedulerPersistParams::RasterOrderOptions;

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k, l;
  RasterOrderOptions raster;
  int swizzle;

  Options():
    help(false),
    m(1024), n(1024), k(1024), l(1),
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
    cmd.get_cmd_line_argument("l", l);
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

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

template <typename Element, typename Layout>
bool initialize_tensor(
    mutlass::TensorView<Element, Layout> view,
    mutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

  if (dist_kind == mutlass::Distribution::Uniform) {
    Element scope_max, scope_min;

    int bits_input = mutlass::sizeof_bits<Element>::value;
    int bits_output = mutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = Element(2);
      scope_min = Element(0);
    } else if (bits_input <= 8) {
      scope_max = Element(2);
      scope_min = Element(-2);
    } else if (bits_output == 16) {
      scope_max = Element(5);
      scope_min = Element(-5);
    } else {
      scope_max = Element(8);
      scope_min = Element(-8);
    }

    mutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min, 0);
  }
  else if (dist_kind == mutlass::Distribution::AllZeros) {
    mutlass::reference::host::TensorFill(view);
  }
  else if (dist_kind == mutlass::Distribution::Identity) {
    mutlass::reference::host::TensorFillIdentity(view);
  }
  else if (dist_kind == mutlass::Distribution::Gaussian) {
    mutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }
  else if (dist_kind == mutlass::Distribution::Sequential) {
    mutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  }
  else {
    throw std::runtime_error("Not implementated.");
  }
  return true;
}

template <typename Element, typename Layout>
bool initialize_scale_tensor(
    mutlass::TensorView<Element, Layout> view,
    mutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

  if (dist_kind == mutlass::Distribution::Uniform) {
    double scope_max, scope_min;
    scope_max = 1e-4;
    scope_min = 1e-5;
    mutlass::reference::host::TensorFillRandomUniform(
      view, seed, scope_max, scope_min);
  }
  else if (dist_kind == mutlass::Distribution::AllZeros) {
    mutlass::reference::host::TensorFill(view);
  }
  else if (dist_kind == mutlass::Distribution::Identity) {
    mutlass::reference::host::TensorFillIdentity(view);
  }
  else if (dist_kind == mutlass::Distribution::Gaussian) {
    mutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }
  else if (dist_kind == mutlass::Distribution::Sequential) {
    mutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  }
  else {
    throw std::runtime_error("Not implementated.");
  }
  return true;
}


/// Initialize operands to be used in the GEMM and reference GEMM
template <class GroupScaleConfig>
void initialize(const Options &options) {
  stride_A = mutlass::make_mute_packed_stride(StrideA{}, {options.m, options.k, options.l});
  stride_B = mutlass::make_mute_packed_stride(StrideB{}, {options.n, options.k, options.l});
  stride_C = mutlass::make_mute_packed_stride(StrideC{}, {options.m, options.n, options.l});
  stride_D = mutlass::make_mute_packed_stride(StrideD{}, {options.m, options.n, options.l});

  using TileShape = typename GroupScaleConfig::TileShape;
  constexpr int ScaleMsPerTile = GroupScaleConfig::ScaleMsPerTile;
  constexpr int ScaleNsPerTile = GroupScaleConfig::ScaleNsPerTile;
  constexpr int ScalePromotionInterleave = GroupScaleConfig::ScalePromotionInterleave;

  // Get Group Scaling tensor shapes
  auto gemm_problem_shape = mute::make_shape(options.m, options.n, options.k);
  auto scale_shape = shape(get<1>(mute::zipped_divide(mute::make_layout(gemm_problem_shape), TileShape{})));
  auto groupscale_m = mute::get<0>(scale_shape) * ScaleMsPerTile;
  auto groupscale_n = mute::get<1>(scale_shape) * ScaleNsPerTile;
  auto groupscale_k = mute::get<2>(scale_shape) / ScalePromotionInterleave;


  auto a_coord = mutlass::make_Coord(options.m * options.l, options.k);
  auto c_coord = mutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = mutlass::make_Coord(options.k, options.n * options.l);
  auto groupscale_a_coord = mutlass::make_Coord(groupscale_m * options.l, groupscale_k);
  auto groupscale_b_coord = mutlass::make_Coord(groupscale_n * options.l, groupscale_k);


  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  blockscale_tensor_A.resize(groupscale_a_coord);
  blockscale_tensor_B.resize(groupscale_b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

  mutlass::Distribution::Kind dist_A = mutlass::Distribution::Uniform;
  mutlass::Distribution::Kind dist_B = mutlass::Distribution::Uniform;
  mutlass::Distribution::Kind dist_C = mutlass::Distribution::Uniform;
  mutlass::Distribution::Kind dist_scaleA = mutlass::Distribution::Uniform;
  mutlass::Distribution::Kind dist_scaleB = mutlass::Distribution::Uniform;

  initialize_tensor(tensor_A.host_view(), dist_A, seed + 2022);
  initialize_tensor(tensor_B.host_view(), dist_B, seed + 2023);
  initialize_tensor(tensor_C.host_view(), dist_C, seed + 2024);
  initialize_scale_tensor(blockscale_tensor_A.host_view(), dist_scaleA, seed + 2025);
  initialize_scale_tensor(blockscale_tensor_B.host_view(), dist_scaleB, seed + 2026);

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();
  blockscale_tensor_A.sync_device();
  blockscale_tensor_B.sync_device();

}

/// Populates a Gemm::Arguments structure from the given commandline options
template <class GemmArguments>
GemmArguments args_from_options(const Options &options)
{
  GemmArguments arguments{
    mutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k},
    {tensor_A.device_data(), stride_A, tensor_B.device_data(), stride_B, blockscale_tensor_A.device_data(), blockscale_tensor_B.device_data()},
    {{options.alpha, options.beta}, tensor_C.device_data(), stride_C, tensor_D.device_data(), stride_D}
  };

  arguments.scheduler.raster_order = options.raster;
  arguments.scheduler.swizzle_size = options.swizzle;

  return arguments;
}

template <typename GroupScaleConfig>
bool verify(Options const& options) {
  using TileShape = typename GroupScaleConfig::TileShape;
  constexpr int ScaleGranularityM = GroupScaleConfig::ScaleGranularityM;
  constexpr int ScaleGranularityN = GroupScaleConfig::ScaleGranularityN;
  constexpr int ScaleGranularityK = GroupScaleConfig::ScaleGranularityK;

  constexpr int ScaleMsPerTile           = GroupScaleConfig::ScaleMsPerTile;
  constexpr int ScaleNsPerTile           = GroupScaleConfig::ScaleNsPerTile;
  constexpr int ScalePromotionInterleave = GroupScaleConfig::ScalePromotionInterleave;

  // Get Group Scaling tensor shapes
  auto gemm_problem_shape = mute::make_shape(options.m, options.n, options.k);
  auto scale_shape = shape(get<1>(mute::zipped_divide(mute::make_layout(gemm_problem_shape), TileShape{})));
  auto groupscale_m = mute::get<0>(scale_shape) * ScaleMsPerTile;
  auto groupscale_n = mute::get<1>(scale_shape) * ScaleNsPerTile;
  auto groupscale_k = mute::get<2>(scale_shape) / ScalePromotionInterleave;


  auto A = mute::make_tensor(tensor_A.host_data(),
                             mute::make_layout(
                               mute::make_shape(options.m, options.k, options.l),
                               stride_A
                             ));
  auto B = mute::make_tensor(tensor_B.host_data(),
                             mute::make_layout(
                               mute::make_shape(options.n, options.k, options.l),
                               stride_B
                             ));
  auto C = mute::make_tensor(tensor_C.host_data(),
                             mute::make_layout(
                               mute::make_shape(options.m, options.n, options.l),
                               stride_C
                             ));
  auto D = mute::make_tensor(tensor_ref_D.host_data(),
                             mute::make_layout(
                               mute::make_shape(options.m, options.n, options.l),
                               stride_D
                             ));

  auto blockscale_A = mute::make_tensor(blockscale_tensor_A.host_data(),
                                        mute::make_layout(
                                          mute::make_shape(groupscale_m, groupscale_k, options.l),
                                          mute::make_stride(_1{}, groupscale_m, groupscale_m * groupscale_k)
                                        ));
  auto blockscale_B = mute::make_tensor(blockscale_tensor_B.host_data(),
                                        mute::make_layout(
                                          mute::make_shape(groupscale_n, groupscale_k, options.l),
                                          mute::make_stride(_1{}, groupscale_n, groupscale_n * groupscale_k)
                                        ));


  using ScaleTileShape = mute::Shape<mute::tuple_element_t<0, TileShape>,
                                     mute::tuple_element_t<1, TileShape>,
                                     Int<ScaleGranularityK>>;
  using unused_t = decltype(D);

  mutlass::reference::host::GettMainloopParams<ElementAccumulator,
                                               decltype(A), decltype(B),
                                               decltype(blockscale_A), decltype(blockscale_B),
                                               ScaleTileShape>
                                mainloop_params {
                                  A, B,
                                  blockscale_A, blockscale_B
                                };

  mutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D)>
    epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;

  // get reference result
  mutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  tensor_D.sync_host();
  bool passed = mutlass::reference::host::TensorRelativelyEquals(tensor_ref_D.host_view(), tensor_D.host_view(), ElementD(1e-3), ElementD(1e-3));

  if (false) {
    std::cout << "tensor_ref_D.host_view() {" << std::endl
              << tensor_ref_D.host_view() << std::endl
              << "}"  << std::endl;
    std::cout << "tensor_D.host_view() {" << std::endl
              << tensor_D.host_view() << std::endl
              << "}"  << std::endl;
  }

  return passed;
}

template <typename GroupScaleConfig, typename Gemm>
int run(Options const& options) {

  initialize<GroupScaleConfig>(options);

  // Instantiate MUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<typename Gemm::Arguments>(options);

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

  bool passed = verify<GroupScaleConfig>(options);

  std::cout << "  Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

  if (!passed) {
    exit(-1);
  }

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
    double avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    double gflops = options.gflops(avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << gflops << std::endl;
  }

  return 0;
}


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
    //options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate MUTLASS kernels
  //
  run<GroupScale1x128x128Config, GroupScale1x128x128Gemm::Gemm>(options);


  tensor_A.reset();
  tensor_B.reset();
  tensor_C.reset();
  blockscale_tensor_A.reset();
  blockscale_tensor_B.reset();
  tensor_D.reset();
  tensor_ref_D.reset();
  return 0;
}
