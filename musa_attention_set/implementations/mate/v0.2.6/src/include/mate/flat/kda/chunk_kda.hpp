#pragma once

#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/collective/chunk_kda_collective_tme_warpspecialized.hpp"
#include "mate/flat/kda/kernel/chunk_kda_kernel_tme_warpspecialized.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

template <class Element_,
          class StateElement_,
          class CuSeqlensElement_,
          class TileShape_,
          class StrideQ_,
          class StrideK_,
          class StrideV_,
          class StrideG_,
          class StrideO_,
          class... Options_>
struct ChunkKdaBuilder {
  using Element            = Element_;
  using StateElement       = StateElement_;
  using CuSeqlensElement   = CuSeqlensElement_;
  using TileShape          = TileShape_;
  using StrideQ            = StrideQ_;
  using StrideK            = StrideK_;
  using StrideV            = StrideV_;
  using StrideG            = StrideG_;
  using StrideO            = StrideO_;
  using CollectiveMainloop = ChunkKdaCollectiveTmeWarpSpecialized<Element,
                                                                  StateElement,
                                                                  TileShape,
                                                                  StrideQ,
                                                                  StrideK,
                                                                  StrideV,
                                                                  StrideG,
                                                                  StrideO,
                                                                  Options_...>;
  using TileScheduler_     = ChunkKdaTileScheduler<TileShape, CuSeqlensElement, Options_...>;
  using Kernel             = ChunkKdaKernel<CollectiveMainloop, TileScheduler_>;
};

}  // namespace mate::flat::kda
