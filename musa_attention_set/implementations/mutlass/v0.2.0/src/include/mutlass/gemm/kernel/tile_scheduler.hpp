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
    \brief Utilities for selecting default tile schedulers
*/

#include "mutlass/detail/dependent_false.hpp"
#include "mutlass/gemm/kernel/static_tile_scheduler.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm {

////////////////////////////////////////////////////////////////////////////////

//
// Tags for specifying tile schedulers
//

struct PersistentScheduler { };

struct StreamKScheduler { };

struct GroupScheduler { }; // Only used for Grouped GEMMs

struct DefaultScheduler { };

////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm

////////////////////////////////////////////////////////////////////////////////

namespace mutlass::gemm::kernel::detail {

//
// Selectors mapping tile scheduler tag and arch tag to a tile scheduler class
//

template <
  class TileSchedulerTag,
  class ArchTag,
  class TileShape,
  class ClusterShape
  , class ProblemShapeType = void
>
struct TileSchedulerSelector {
  static_assert(mutlass::detail::dependent_false<ArchTag>,
      "Could not select a tile scheduler for given parameters.");
};

template <
  class TileShape,
  class ClusterShape,
  class ProblemShapeType
>
struct TileSchedulerSelector <
  void,
  arch::Mp31,
  TileShape,
  ClusterShape,
  ProblemShapeType
> {
  using Scheduler = typename TileSchedulerSelector<
    DefaultScheduler,
    arch::Mp31,
    TileShape,
    ClusterShape,
    ProblemShapeType
  >::Scheduler;
};

template <
  class TileShape,
  class ClusterShape,
  class ProblemShapeType
>
struct TileSchedulerSelector <
  DefaultScheduler,
  arch::Mp31,
  TileShape,
  ClusterShape,
  ProblemShapeType
> {
  using Scheduler = StaticTileScheduler;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace mutlass::gemm::kernel::detail

////////////////////////////////////////////////////////////////////////////////
