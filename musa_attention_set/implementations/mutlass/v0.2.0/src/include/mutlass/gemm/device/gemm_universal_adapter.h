/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*!
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and
    batched array variants.
*/

#pragma once

// common
#include "mutlass/mutlass.h"
#include "mutlass/device_kernel.h"
#include "mutlass/gemm/gemm.h"
#include "mutlass/detail/layout.hpp"
#include "mutlass/detail/mma.hpp"
#include "mutlass/musa_host_adapter.hpp"

#if !defined(__MUSACC_RTC__)
//#include "mutlass/cluster_launch.hpp"
#include "mutlass/trace.h"
#endif // !defined(__MUSACC_RTC__)

#include "mutlass/gemm/threadblock/threadblock_swizzle.h" // placeholder for thread block swizzle

// 3.x
#include "mutlass/arch/mma.h" // mutlass::arch::OpMultiplyAdd
#include "mutlass/gemm/kernel/gemm_universal.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::device {

////////////////////////////////////////////////////////////////////////////////

/*!
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type mutlass::gemm::kernel::Gemm or mutlass::gemm::kernel::GemmUniversal.

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behaviour might
  differ between the two specializations.
*/
template <class GemmKernel_, class Enable = void>
class GemmUniversalAdapter;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// MUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <class GemmKernel_>
class GemmUniversalAdapter<
  GemmKernel_,
  mute::enable_if_t<gemm::detail::IsMutlass3GemmKernel<GemmKernel_>::value>>
{
public:
  using GemmKernel = GemmKernel_;
  using TileShape = typename GemmKernel::TileShape;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  // using ElementC = typename GemmKernel::ElementC;
  // using ElementC = mute::conditional_t<mute::is_void_v<GemmKernel::ElementC>, ElementD, GemmKernel::ElementC>;
  using ElementD = typename GemmKernel::ElementD;
  using ElementC = mute::conditional_t<mute::is_void_v<typename GemmKernel::ElementC>, ElementD, typename GemmKernel::ElementC>;
  using ElementAccumulator = typename GemmKernel::ElementAccumulator;
  using DispatchPolicy = typename GemmKernel::DispatchPolicy;
  using CollectiveMainloop = typename GemmKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;

  // Map back to 2.x type as best as possible
  using LayoutA = gemm::detail::StrideToLayoutTagA_t<typename GemmKernel::StrideA>;
  using LayoutB = gemm::detail::StrideToLayoutTagB_t<typename GemmKernel::StrideB>;
  using LayoutC = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideC>;
  using LayoutD = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideD>;

  static bool const kEnableMusaHostAdapter = MUTLASS_ENABLE_MUSA_HOST_ADAPTER;

  static ComplexTransform const kTransformA = mute::is_same_v<typename GemmKernel::CollectiveMainloop::TransformA, mute::conjugate> ?
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;
  static ComplexTransform const kTransformB = mute::is_same_v<typename GemmKernel::CollectiveMainloop::TransformB, mute::conjugate> ?
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;

  // Legacy: Assume MultiplyAdd only since we do not use this tag type in 3.0
  using MathOperator = mutlass::arch::OpMultiplyAdd;

  using OperatorClass = mutlass::detail::get_operator_class_t<typename CollectiveMainloop::TiledMma>;

  using ArchTag = typename GemmKernel::ArchTag;

  // NOTE: Assume identity swizzle for now
  using ThreadblockSwizzle = mutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // Assume TiledMma's ShapeMNK is the same as 2.x's ThreadblockShape
  using ThreadblockShape = mutlass::gemm::GemmShape<
      mute::size<0>(TileShape{}),
      mute::size<1>(TileShape{}),
      mute::size<2>(TileShape{})>;

  using ClusterShape = mutlass::gemm::GemmShape<
      mute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      mute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      mute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{})>;

  // Instruction shape is easy too, since we get that directly from our TiledMma's atom shape
  using InstructionShape = mutlass::gemm::GemmShape<
      mute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      mute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      mute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{})>;

  // Legacy: provide a correct warp count, but no reliable warp shape
  static int const kThreadCount = GemmKernel::MaxThreadsPerBlock;

  // Warp shape is not a primary API type in 3.x
  // But we can best approximate it by inspecting the TiledMma
  // For this, we make the assumption that we always have 4 warps along M, and rest along N, none along K
  // We also always round up the warp count to 4 if the tiled mma is smaller than 128 threads
  static constexpr int WarpsInMma = mute::max(4, MUTE_STATIC_V(mute::size(typename GemmKernel::TiledMma{})) / 32);
  static constexpr int WarpsInMmaM = 4;
  static constexpr int WarpsInMmaN = mute::ceil_div(WarpsInMma, WarpsInMmaM);
  using WarpCount = mutlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;
  using WarpShape = mutlass::gemm::GemmShape<
      MUTE_STATIC_V(mute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
      MUTE_STATIC_V(mute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
      MUTE_STATIC_V(mute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;

  static int constexpr kStages = CollectiveMainloop::DispatchPolicy::Stages;

  // Inspect TiledCopy for A and B to compute the alignment size
  static int constexpr kAlignmentA = mutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyA, ElementA, typename CollectiveMainloop::TiledMma::ValTypeA>();
  static int constexpr kAlignmentB = mutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyB, ElementB, typename CollectiveMainloop::TiledMma::ValTypeB>();
  static int constexpr kAlignmentC = mutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyC, ElementC>();
  static int constexpr kAlignmentD = mutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyD, ElementD>();

  using EpilogueOutputOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  // Split-K preserves splits that are 128b aligned
  static int constexpr kSplitKAlignment = mute::max(
      128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  /// Argument structure: User API
  using Arguments = typename GemmKernel::Arguments;
  /// Argument structure: Kernel API
  using Params = typename GemmKernel::Params;

private:

  /// Kernel API parameters object
  Params params_;

public:

  /// Access the Params structure
  Params const& params() const {
    return params_;
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    if (GemmKernel::can_implement(args)) {
      return Status::kSuccess;
    }
    else {
      return Status::kInvalid;
    }
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    if (args.mode == GemmUniversalMode::kGemmSplitKParallel) {
      workspace_bytes += sizeof(int) * size_t(mute::size<0>(TileShape{})) * size_t(mute::size<1>(TileShape{}));
    }

    MUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    workspace_bytes += GemmKernel::get_workspace_size(args);
    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Arguments const& args, void* workspace = nullptr) {
    auto tmp_params = GemmKernel::to_underlying_arguments(args, workspace);
    return GemmKernel::get_grid_shape(tmp_params);
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Params const& params) {
    return GemmKernel::get_grid_shape(params);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    MUTLASS_TRACE_HOST("GemmUniversal::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = GemmKernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    musaError_t result;
    if (smem_size >= (48 << 10)) {
      MUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = musaFuncSetAttribute(
          device_kernel<GemmKernel>,
          musaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (musaSuccess != result) {
        result = musaGetLastError(); // to clear the error bit
        MUTLASS_TRACE_HOST(
          "  musaFuncSetAttribute() returned error: "
          << musaGetErrorString(result));
        return -1;
      }
    }

    // query occupancy after setting smem size
    result = musaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        device_kernel<GemmKernel>,
        GemmKernel::MaxThreadsPerBlock,
        smem_size);

    if (musaSuccess != result) {
      result = musaGetLastError(); // to clear the error bit
      MUTLASS_TRACE_HOST(
        "  musaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << musaGetErrorString(result));
      return -1;
    }

    MUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Initializes GEMM state from arguments.
  Status
  initialize(
    Arguments const& args,
    void* workspace = nullptr,
    musaStream_t stream = nullptr,
    MusaHostAdapter* musa_adapter = nullptr) {

    MUTLASS_TRACE_HOST("GemmUniversal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize the workspace
    Status status = GemmKernel::initialize_workspace(args, workspace, stream, musa_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    // Initialize the Params structure
    params_ = GemmKernel::to_underlying_arguments(args, workspace);
    // Don't set the function attributes - require the MusaHostAdapter to set it.
    if constexpr (kEnableMusaHostAdapter) {
      MUTLASS_ASSERT(musa_adapter);
      return Status::kSuccess;
    }
    else {
      //
      // Account for dynamic smem capacity if needed
      //
      int smem_size = GemmKernel::SharedStorageSize;

      MUTLASS_ASSERT(musa_adapter == nullptr);

      if (smem_size >= (48 << 10)) {
        MUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
        musaError_t result = musaFuncSetAttribute(
            device_kernel<GemmKernel>,
            musaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
        if (musaSuccess != result) {
          result = musaGetLastError(); // to clear the error bit
          MUTLASS_TRACE_HOST("  musaFuncSetAttribute() returned error: " << musaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }
    }
    return Status::kSuccess;
  }

  /// Update API is preserved in 3.0, but does not guarantee a lightweight update of params.
  Status
  update(Arguments const& args, void* workspace = nullptr) {
    MUTLASS_TRACE_HOST("GemmUniversal()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);
    if (workspace_bytes > 0 && nullptr == workspace) {
      return Status::kErrorWorkspaceNull;
    }

    params_ = GemmKernel::to_underlying_arguments(args, workspace);
    return Status::kSuccess;
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling GemmKernel::to_underling_arguments()
  static Status
  run(Params& params,
      musaStream_t stream = nullptr,
      MusaHostAdapter *musa_adapter = nullptr) {
    MUTLASS_TRACE_HOST("GemmUniversal::run()");
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = GemmKernel::SharedStorageSize;

    Status launch_result{ Status::kSuccess };
    // Use extended launch API only for mainloops that use it
    launch_result = Status::kSuccess;
    if constexpr (kEnableMusaHostAdapter) {
      MUTLASS_ASSERT(musa_adapter);
      if (musa_adapter) {
        void* kernel_params[] = {&params};

        launch_result = musa_adapter->launch(
          grid, block, smem_size, stream, kernel_params, 0
        );

      }
      else {
        return Status::kErrorInternal;
      }
    }
    else {
      MUTLASS_ASSERT(musa_adapter == nullptr);
      device_kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params);
    }

    musaError_t result = musaGetLastError();
    if (musaSuccess == result && Status::kSuccess == launch_result) {
      return Status::kSuccess;
    }
    else {
      MUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  run(
    Arguments const& args,
    void* workspace = nullptr,
    musaStream_t stream = nullptr,
    MusaHostAdapter *musa_adapter = nullptr
  ) {
    Status status = initialize(args, workspace, stream, musa_adapter);

    if (Status::kSuccess == status) {
      status = run(params_, stream, musa_adapter);
    }
    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  operator()(
    Arguments const& args,
    void* workspace = nullptr,
    musaStream_t stream = nullptr,
    MusaHostAdapter *musa_adapter = nullptr) {
    return run(args, workspace, stream, musa_adapter);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(musaStream_t stream = nullptr, MusaHostAdapter *musa_adapter = nullptr) {
    return run(params_, stream, musa_adapter);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  operator()(musaStream_t stream = nullptr, MusaHostAdapter *musa_adapter = nullptr) {
    return run(params_, stream, musa_adapter);
  }
};

} // namespace mutlass::gemm::device

////////////////////////////////////////////////////////////////////////////////
