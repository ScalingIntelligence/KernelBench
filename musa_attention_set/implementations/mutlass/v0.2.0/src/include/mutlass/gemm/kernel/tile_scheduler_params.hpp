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

/*! \file
    \brief Parameters structures for persistent tile schedulers
*/

#include "mutlass/coord.h"
#include "mutlass/kernel_hardware_info.h"
#include "mutlass/workspace.h"
#include "mutlass/platform/platform.h"
#include "mutlass/fast_math.h"
#include "mutlass/gemm_coord.h"
////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::kernel::detail {

////////////////////////////////////////////////////////////////////////////////

struct TileSchedulerDefaultParams {

  enum class RasterOrder {
    AlongM,
    AlongN
  };

  enum class RasterOrderOptions {
    Heuristic,
    AlongM,
    AlongN
  };


  FastDivmod divmod_macro_tile_x_;
  FastDivmod divmod_macro_tile_size_;
  int32_t swizzle_size_ = 0;
  RasterOrder raster_order_ = RasterOrder::AlongN;
  dim3 problem_blocks_{0, 0, 0};

  // Initializes members. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  void
  initialize(
    BatchedGemmCoord problem_shape,
    GemmCoord tile_shape,
    KernelHardwareInfo const& hw_info,
    int swizzle_size,
    RasterOrderOptions raster_order_option
  ) {
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape, tile_shape);
    return initialize(
      problem_blocks,
      hw_info,
      swizzle_size,
      raster_order_option
    );
  }

  // Version of initialize that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using MuTe algebra for calculating tile shapes is easiest.
  void
  initialize(
    dim3 problem_blocks,
    KernelHardwareInfo const& hw_info,
    int swizzle_size,
    RasterOrderOptions raster_order_option
  ) {
    MUTLASS_UNUSED(hw_info);
    RasterOrder raster_order = get_rasterization_order(
      problem_blocks.y,
      problem_blocks.x,
      raster_order_option
    );

    swizzle_size_ = swizzle_size;
    raster_order_ = raster_order;
    problem_blocks_ = problem_blocks;
    if (raster_order_ == RasterOrder::AlongN) {
      divmod_macro_tile_size_ = FastDivmod(swizzle_size_ * problem_blocks.x);
    } else {
      divmod_macro_tile_size_ = FastDivmod(swizzle_size_ * problem_blocks.y);
    }
    divmod_macro_tile_x_ = FastDivmod(swizzle_size_);
  }

  // Version of get_grid_shape that takes in as input the number of CTAs in the M and N and L dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using MuTe algebra for calculating tile shapes is easiest.
  MUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    dim3 problem_blocks,
    KernelHardwareInfo const& hw_info,
    int swizzle_size,
    RasterOrderOptions raster_order_option
  ) {

    MUTLASS_UNUSED(hw_info);
    RasterOrder raster_order = get_rasterization_order(
      problem_blocks.x,
      problem_blocks.y,
      raster_order_option
    );

    return dim3(problem_blocks.x * problem_blocks.y, 1, problem_blocks.z);
  }

  MUTLASS_HOST_DEVICE
  static RasterOrder
  get_rasterization_order(
    uint32_t tiles_m,
    uint32_t tiles_n,
    RasterOrderOptions raster_order_option
  ) {

    if (raster_order_option == RasterOrderOptions::Heuristic) {
      if (tiles_n > tiles_m) {
        return RasterOrder::AlongM;
      }
      else {
        return RasterOrder::AlongN;
      }
    }
    else if (raster_order_option == RasterOrderOptions::AlongN) {
        return RasterOrder::AlongN;
    }
    else {
      return RasterOrder::AlongM;
    }
  }

  // Get the number of CTA tiles in this problem. This variant of the method should only be used when
  // problem_shape and tile_shape contain modes of only rank 1.
  MUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(BatchedGemmCoord problem_shape, GemmCoord cta_shape) {
    auto cta_m = (problem_shape.m() + cta_shape.m() - 1) / cta_shape.m();
    auto cta_n = (problem_shape.n() + cta_shape.n() - 1) / cta_shape.n();

    return get_tiled_cta_shape_mnl(problem_shape, cta_m, cta_n);
  }

  // Version of get_tiled_cta_shape_mnl that takes in as input the number of CTAs in the M and N dimensions.
  // This is useful for calculating the tiled shape when a mode of problem and/or CTA shape has rank > 1,
  // for which using MuTe algebra for calculating tile shapes is easiest.
  MUTLASS_HOST_DEVICE
  static dim3
  get_tiled_cta_shape_mnl(BatchedGemmCoord problem_shape, uint32_t cta_m, uint32_t cta_n) {
    return {
      static_cast<uint32_t>(cta_m),
      static_cast<uint32_t>(cta_n),
      static_cast<uint32_t>(problem_shape.batch())
    };
  }
};

////////////////////////////////////////////////////////////////////////////////
} // namespace mutlass::gemm::kernel::detail

////////////////////////////////////////////////////////////////////////////////
