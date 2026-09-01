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
#pragma once

#include "mutlass/fast_math.h"
#include "mutlass/gemm_coord.hpp"
#include "mutlass/kernel_hardware_info.hpp"
#include "mutlass/gemm/kernel/tile_scheduler_params.hpp"
#include "mute/layout.hpp"
#include "mute/tensor.hpp"
#include "mutlass/pipeline/pipeline.hpp"

namespace mutlass::gemm::kernel::detail {

using namespace mute;
///////////////////////////////////////////////////////////////////////////////

class PersistentTileScheduler {
  
private:
  uint64_t current_work_linear_idx_;
public:
  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    bool is_valid_tile = false;
    MUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      return is_valid_tile;
    }

    MUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, false};
    }
  };

  

  using Params = TileSchedulerPersistParams;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;

public:
  struct Arguments {
    int swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::AlongN;
  };

  template <class ProblemShapeMNKL, class TileShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      [[maybe_unused]] KernelHardwareInfo const& hw_info,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace=nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {

    static_assert(mute::is_static<TileShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape);

    Params params;
    params.initialize(
      problem_blocks,
      hw_info,
      arguments.swizzle_size,
      arguments.raster_order
    );
    return params;
  }

  MUTLASS_HOST_DEVICE
  static bool
  can_implement(Arguments const& args) {
    return args.swizzle_size >= 1;
  }

  MUTLASS_HOST_DEVICE
  PersistentTileScheduler() { }

  MUTLASS_DEVICE explicit PersistentTileScheduler(Params const& params_) : scheduler_params(params_) {
    current_work_linear_idx_ = blockIdx.x;
  }

  // Returns the initial work tile info that will be computed over
  MUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info() {
    current_work_linear_idx_ = blockIdx.x;
    return get_current_work(current_work_linear_idx_);
  }

  MUTLASS_DEVICE
  WorkTileInfo
  get_work_tile_info() {
    return get_current_work(current_work_linear_idx_);
  }

  MUTLASS_DEVICE
  WorkTileInfo
  get_current_work(uint64_t current_work_linear_idx) const {
    auto problem_blocks = scheduler_params.problem_blocks_;
    if (current_work_linear_idx >= problem_blocks.x * problem_blocks.y)   //check if the linear_idx is valid
      return WorkTileInfo::invalid_work_tile();
    int macro_tile_x = scheduler_params.swizzle_size_;
    int macro_tile_size, res_tile_x, res_tile_x_quo, macro_bid_max, max_tile_y, mini_x, blockidx_xy, macro_x, macro_res;
    if (scheduler_params.raster_order_ == RasterOrder::AlongN) {
      scheduler_params.divmod_macro_tile_x_.fast_divmod(res_tile_x_quo, res_tile_x, problem_blocks.y);
      macro_bid_max = res_tile_x_quo * macro_tile_x * problem_blocks.x;
      max_tile_y = problem_blocks.x - 1;
    } else {
      scheduler_params.divmod_macro_tile_x_.fast_divmod(res_tile_x_quo, res_tile_x, problem_blocks.x);
      macro_bid_max = res_tile_x_quo * macro_tile_x * problem_blocks.y;
      max_tile_y = problem_blocks.y - 1;
    }
    uint32_t block_idx_x = current_work_linear_idx;
    uint32_t block_idx_y = 1;
    uint32_t block_idx_z = blockIdx.z;
    blockidx_xy = current_work_linear_idx;
    scheduler_params.divmod_macro_tile_size_.fast_divmod(macro_x, macro_res, blockidx_xy);

    if (blockidx_xy >= macro_bid_max) {
      block_idx_y = macro_res / res_tile_x;
      mini_x = macro_res % res_tile_x;
      if (block_idx_y % 2 == 1) mini_x = res_tile_x - 1 - mini_x;
    } else {
      block_idx_y = macro_res / macro_tile_x;
      mini_x = macro_res % macro_tile_x;
      if (block_idx_y % 2 == 1) {
        mini_x = macro_tile_x - 1 - mini_x;
      }
    }
    if (macro_x % 2 == 1) {
      block_idx_y = max_tile_y - block_idx_y;
    }
    block_idx_x = macro_x * macro_tile_x + mini_x;
    if (scheduler_params.raster_order_ == RasterOrder::AlongN) {
      return {static_cast<int32_t>(block_idx_y), static_cast<int32_t>(block_idx_x), static_cast<int32_t>(block_idx_z), true};
    } else {
      return {static_cast<int32_t>(block_idx_x), static_cast<int32_t>(block_idx_y), static_cast<int32_t>(block_idx_z), true};
    }
  }
  MUTLASS_DEVICE
  void
  advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += gridDim.x * gridDim.y * advance_count;
  }

  MUTLASS_DEVICE
  static bool
  continue_current_work(WorkTileInfo&) {
    return false;
  }


  // Given the inputs, computes the total number of output blocks over which this problem will compute.
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template<class ProblemShapeMNKL, class BlockShape>
  MUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl, BlockShape cta_shape) {
    auto cta_m = mute::size(mute::ceil_div(mute::shape<0>(problem_shape_mnkl), mute::shape<0>(cta_shape)));
    auto cta_n = mute::size(mute::ceil_div(mute::shape<1>(problem_shape_mnkl), mute::shape<1>(cta_shape)));

    return Params::get_tiled_cta_shape_mnl(
      to_gemm_coord(problem_shape_mnkl),
      cta_m, cta_n
    );
  }

  // Given the inputs, computes the physical grid we should launch.
  template<class ProblemShapeMNKL, class BlockShape>
  MUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
      [[maybe_unused]] Params const& params,
      ProblemShapeMNKL problem_shape_mnk,
      BlockShape cta_shape,
      KernelHardwareInfo hw_info,
      Arguments arguments = Arguments{1, RasterOrderOptions::AlongN},
      bool truncate_by_problem_size=true) {

    auto problem_shape_mnkl = mute::append<4>(problem_shape_mnk, mute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape);

    return Params::get_grid_shape(problem_blocks, hw_info, arguments.swizzle_size, arguments.raster_order);
  }

public:
  // Sink scheduler params as a member
  Params scheduler_params;
}; // end class PersistentTileScheduler

} // namespace mutlass::gemm::kernel::detail
