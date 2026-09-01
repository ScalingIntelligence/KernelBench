#pragma once

#include "mate/attention/msa/collective/mp31_msa_maxscore_collective_tme_warpspecialized.hpp"
#include "mate/attention/msa/kernel/msa_maxscore_kernel_tme_warpspecialized.hpp"
#include "mate/attention/msa/kernel/msa_maxscore_tile_scheduler.hpp"
#include "mate/attention/msa/msa_options.hpp"

namespace mate::attention::msa {

template <class Element_,
          class TileShape_,
          int  HeadRatio_,
          int  QStages_,
          int  KStages_,
          bool ParallelKTiles_,
          class... Options_>
struct MsaMaxScoreBuilder {
  using Element                        = Element_;
  using TileShape                      = TileShape_;
  static constexpr int  HeadRatio      = HeadRatio_;
  static constexpr int  QStages        = QStages_;
  static constexpr int  KStages        = KStages_;
  static constexpr bool ParallelKTiles = ParallelKTiles_;
  using CollectiveMainloop             = collective::
      Mp31MsaMaxScoreCollectiveTmeWarpSpecialized<Element, TileShape, HeadRatio, QStages, KStages, Options_...>;
  using TileScheduler = MsaMaxScoreTileScheduler<TileShape, HeadRatio, ParallelKTiles>;
  using Kernel        = MsaMaxScoreKernelTmeWarpSpecialized<CollectiveMainloop, TileScheduler>;
};

}  // namespace mate::attention::msa
